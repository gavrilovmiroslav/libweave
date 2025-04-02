use std::cell::{RefCell, RefMut};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hasher};
use id_arena::{Arena, Id};
use itertools::Itertools;
use multimap::MultiMap;

/// Type alias for motif ID in libweave's internal arenas.
pub type MotifId = Id<Motif>;
pub type MotifIdx = usize;

/// A **motif** is the smallest building block in libweave.
///
/// A motif is similar to an entity in ECS frameworks, but holds the capacity for structure.
/// Indeed, every motif is a 3-tuple containing an **identity**, **source**, and **target**,
/// all three of which are internally of the type `MotifId`, and externally `usize`. The enum
/// `Motif` separates different categories of motifs by their use-capabilities:
///
/// - **knots** have the form `(id, id, id)` and are most like nodes in graph libraries.
/// - **tethers** have the form `(id, a, id)` where `a` is different from `id`. Tethers are like
///     arrows from a real motif `a` that extends towards `id` without forming an arrow. Tethers
///     are useful for representing outgoing attachments or hierarchies.
/// - **marks** have the form `(id, id, a)` where `a` is different from `id`. Marks are like arrows
///     that go into a motif `id` but don't represent real objects. Marks are useful for representing
///     incoming attachments or properties.
/// - **arrows** have the form `(id, a, b)` where `a` and `b` are specifically different from `id`
///     represent arrows between `a ----id---> b` arrows similar to those in graph libraries.
///
/// Of the four motif types, **knots** and **arrows** form a direct graph-like structure, while
/// marks and tethers are there to describe metadata and hierarchies. However, because all motifs
/// have the same form without any distinction between nodes and arrows in the type system, it's
/// possible (and _very_ useful!) to have arrows _between_ arrows and other motifs. Using this
/// property, complex relations become trivial to model using motifs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Motif {
    Knot,
    Arrow { source: MotifId, target: MotifId },
    Tether { source: MotifId },
    Mark { target: MotifId },
}

/// Weave internals, keeping all allocations and relationships between motifs under wraps.
/// Significant detail: the weave internal registry contains a special motif called `bottom`,
/// which represents an error result in some motif computation (for example, asking for a source
/// of a non-existent motif ID).
#[repr(C)]
#[derive(Clone, Debug)]
pub struct WeaveInternal<'s> {
    pub(crate) motif_bottom: MotifIdx,
    motif_space: Arena<Motif>,
    motif_freelist: VecDeque<MotifId>,
    motif_index: HashMap<MotifIdx, MotifId>,
    motif_conns: MultiMap<(MotifIdx, MotifIdx), MotifIdx>,
    motif_neighbors: MultiMap<MotifIdx, MotifIdx>,
    motif_co_neighbors: MultiMap<MotifIdx, MotifIdx>,
    motif_tethers: MultiMap<MotifIdx, MotifIdx>,
    motif_marks: MultiMap<MotifIdx, MotifIdx>,
    string_space: Arena<&'s str>,
}

impl<'s> Default for WeaveInternal<'s> {
    fn default() -> Self {
        let mut motif_space: Arena<Motif> = Arena::default();
        let bottom = motif_space.alloc(Motif::Knot);
        let mut motif_index = HashMap::default();
        motif_index.insert(bottom.index(), bottom);

        WeaveInternal {
            motif_space,
            motif_bottom: bottom.index(),
            motif_index,
            motif_freelist: Default::default(),
            motif_conns: Default::default(),
            motif_neighbors: Default::default(),
            motif_co_neighbors: Default::default(),
            motif_tethers: Default::default(),
            motif_marks: Default::default(),
            string_space: Default::default(),
        }
    }
}

/// A **cover** is a set of knots that are in some way connected with arrows. The construction of a
/// cover hashes the resulting knots: use this value to quickly check if two knots are transitively
/// connected.
#[derive(Debug, Clone)]
pub struct Cover {
    pub hash: u64,
    pub knots: Vec<MotifIdx>,
}

impl From<Vec<MotifIdx>> for Cover {
    fn from(value: Vec<MotifIdx>) -> Self {
        let mut hasher = DefaultHasher::new();
        for v in &value { hasher.write_usize(*v); }
        Cover { hash: hasher.finish(), knots: value }
    }
}

/// An **embedding** is a pattern-match of a graph within another graph. We then say that one graph is
/// embedded in another, or that there is an isomorphic subgraph within it. The `Embedding` structure
/// saves the relation (in the form of the index of a hoisted arrow) that it is a part of, and offers
/// an `image` containing mappings between nodes of two graphs. Created as part of `find_embeddings`
/// in `Weaveable<W>`.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Embedding {
    pub relation: MotifIdx,
    pub image: Vec<(MotifIdx, MotifIdx)>,
}

/// A context for when searching for embeddings. Embedding searching (or subgraph pattern matching)
/// is a very useful operation that traditionally can be done between two graph covers. This structure
/// keeps all the relevant operational data within it, including the `Weave` in which it's happening,
/// the search embedding process arrow `MotifIdx`, and both `Cover`s.
pub struct SearchEmbeddingContext<'w, 's> {
    pub(crate) weave: Weave<'w, 's>,
    pub(crate) embed: MotifIdx,
    pub(crate) query: Cover,
    pub(crate) data: Cover,
}

pub trait FindEmbeddings {
    fn find_one_embedding(weave: &Weave, embed: MotifIdx, query: Cover, data: Cover) -> Option<Embedding> {
        Self::find_all_embeddings(weave, embed, query, data).first().cloned()
    }

    fn find_all_embeddings(weave: &Weave, embed: MotifIdx, query: Cover, data: Cover) -> Vec<Embedding>;
}

/// The main libweave API, mainly used by being implemented by `WeaveRef`, it defines ways to
/// create, populate, explore, and traverse motifs. As an external API, `Weavable` uses `usize`
/// for all motif IDs instead of `MotifId`s.
pub trait Weaveable<W> {
    /// Creates a new `Weave` instance, although it's hard to tell from the generic type.
    fn create() -> W;

    /// Gets the **source** of a `motif`. The guarantees of `get_source` are the following:
    ///     - if `index` represents a valid `knot` or `mark`, the result will be `index`
    ///     - if `index` represents a valid `arrow` or `tether`, the result will be their `source`
    ///     - if `index` isn't valid, it will return the `bottom` (invalid) ID.
    /// 
    /// # Example
    ///
    ///
    /// ```plaintext
    ///             pre                                         post
    ///  ---------------------------------------------------------------------------
    ///                                                         [a]
    ///        [a]      [b]                            [a] ---->(c)      [b]
    ///   [a] ---->(c)      [b]                    [a] ---->(c)      (d)----> [b]
    /// [a] ---->(c)      (d)----> [b]          [a] ---->(c) >---e---> (d)----> [b]
    ///
    /// [a] ---->(c) >---e---> (d)----> [b]                source(a) = a
    /// ^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                source(c) = a
    ///  ^^^^^^^^^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                source(e) = c
    ///          ^^^^^^^^^^^^^^^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                source(d) = d
    ///                        ^^^^^^^^^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                source(b) = b
    ///                                 ^^^
    /// ```
    /// 
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_tether(a).unwrap();
    /// let d = weave.new_mark(b).unwrap();
    /// let e = weave.new_arrow(c, d).unwrap();
    /// assert_eq!(weave.get_source(a), a);
    /// assert_eq!(weave.get_source(c), a);
    /// assert_eq!(weave.get_source(e), c);
    /// assert_eq!(weave.get_source(d), d);
    /// assert_eq!(weave.get_source(b), b);
    /// ```
    fn get_source(&self, index: MotifIdx) -> MotifIdx;

    /// Gets the n-th degree **source** of a `motif`. This is equivalent to calling `get_source`
    /// `n` times in a sequence, passing the result of the previous query as the argument.
    ///
    /// It holds that: get_source_nth(e, 2) = get_source(get_source(e))
    fn get_source_nth(&self, index: MotifIdx, degree: usize) -> MotifIdx;

    /// Gets the **target** of a `motif`. The guarantees of `get_target` are the following:
    ///     - if `index` represents a valid `knot` or `tether`, the result will be `index`
    ///     - if `index` represents a valid `arrow` or `mark`, the result will be their `target`
    ///     - if `index` isn't valid, it will return the `bottom` (invalid) ID.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///             pre                                         post
    ///  ---------------------------------------------------------------------------
    ///                                                         [a]
    ///        [a]      [b]                            [a] ---->(c)      [b]
    ///   [a] ---->(c)      [b]                    [a] ---->(c)      (d)----> [b]
    /// [a] ---->(c)      (d)----> [b]          [a] ---->(c) >---e---> (d)----> [b]
    ///
    /// [a] ---->(c) >---e---> (d)----> [b]                target(a) = a
    /// ^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                target(c) = c
    ///   ^^^^^^^^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                target(e) = d
    ///          ^^^^^^^^^^^^^^^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                target(d) = d
    ///                        ^^^^^^^^^^^
    /// [a] ---->(c) >---e---> (d)----> [b]                target(b) = b
    ///                                 ^^^
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_tether(a).unwrap();
    /// let d = weave.new_mark(b).unwrap();
    /// let e = weave.new_arrow(c, d).unwrap();
    /// assert_eq!(weave.get_target(a), a);
    /// assert_eq!(weave.get_target(c), c);
    /// assert_eq!(weave.get_target(e), d);
    /// assert_eq!(weave.get_target(d), b);
    /// assert_eq!(weave.get_target(b), b);
    /// ```
    fn get_target(&self, index: MotifIdx) -> MotifIdx;

    /// Gets the n-th degree **target** of a `motif`. This is equivalent to calling `get_target`
    /// `n` times in a sequence, passing the result of the previous query as the argument.
    ///
    /// It holds that: get_target_nth(e, 2) = get_target(get_target(e))
    fn get_target_nth(&self, index: MotifIdx, degree: usize) -> MotifIdx;

    /// Creates a new knot (node-like motif) and returns its index as the result.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  -----------------------------------------------
    ///                                       [a]
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// assert_ne!(a, weave.bottom());
    /// assert!(weave.is_knot(a).unwrap_or(false));
    /// ```
    fn new_knot(&self) -> MotifIdx;

    /// Creates a new arrow (arrow-like motif) and returns an option with its index as the result,
    /// or `None` if either the `source` or `target` motifs are malformed. Can be easily defaulted
    /// to `usize` by using `.unwrap_or(weave.bottom())`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  -----------------------------------------------
    ///        [a]     [b]               [a]>--c-->[b]
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_source(c), a);
    /// assert_eq!(weave.get_target(c), b);
    /// assert!(weave.is_arrow(c).unwrap_or(false));
    /// ```
    fn new_arrow(&self, source: MotifIdx, target: MotifIdx) -> Option<MotifIdx>;

    /// Creates a new tether (arrow-like extension) and returns an option with its index as a result,
    /// or `None` if the `source` motif is malformed. Can be easily defaulted to `usize` by using
    /// `.unwrap_or(weave.bottom())`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  -----------------------------------------------
    ///            [a]                    [a] ---->(b)
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_tether(a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_source(b), a);
    /// assert_eq!(weave.get_target(b), b);
    /// assert!(weave.is_tether(b).unwrap_or(false));
    /// ```
    fn new_tether(&self, source: MotifIdx) -> Option<MotifIdx>;

    /// Creates a new mark (node-like extension) and returns an option with its index as a result,
    /// or `None` if the `target` motif is malformed. Can be easily defaulted to `usize` by using
    /// `.unwrap_or(weave.bottom())`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  -----------------------------------------------
    ///            [a]                    [a] <----(b)
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_mark(a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_source(b), b);
    /// assert_eq!(weave.get_target(a), a);
    /// assert!(weave.is_mark(b).unwrap_or(false));
    /// ```
    fn new_mark(&self, target: MotifIdx) -> Option<MotifIdx>;

    /// Deletes the `target` motif and any other motif it might have been upholding. A motif is
    /// considered to be _upholding_ another motif if its existence hinges on the existence of the
    /// first one. Arrows are upheld by their source and target, marks and tethers are upheld by
    /// their source, and their target, respectively. Deleting a single motif might delete more than
    /// just that, depending on the structure.
    ///
    /// # Example
    /// ```plaintext
    ///            pre                        post
    ///  -----------------------------------------------
    ///                                       [a]
    ///            [a]                    [a] <---(b)
    ///        [a] <---(b)
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_mark(a).unwrap_or(weave.bottom());
    /// weave.delete(a);
    /// assert!(!weave.exists(a));
    /// assert!(!weave.exists(b));
    /// ```
    fn delete(&self, target: MotifIdx) -> bool;

    /// Returns `true` if the `MotifId` has been allocated in storage, and has not been deleted.
    /// Deleting a motif frees the `Motif` for later re-use, and we consider it non-existing.
    fn exists(&self, index: MotifIdx) -> bool;

    /// Identifies the underlying `MotifId` for the given `index`. Mostly internal use.
    fn identify(&self, index: MotifIdx) -> MotifId;

    /// Identifies the underlying `MotifId` for the given `index` but instead of returning
    /// the `bottom` motif, it returns `None` when `index` doesn't exist. Mostly internal use.
    fn opt_identify(&self, index: MotifIdx) -> Option<MotifId>;

    /// Returns an option with `true` if the motif under `index` is a knot. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_knot(&self, index: MotifIdx) -> Option<bool>;

    /// Returns an option with `true` if the motif under `index` is an arrow. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_arrow(&self, index: MotifIdx) -> Option<bool>;

    /// Returns an option with `true` if the motif under `index` is a tether. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_tether(&self, index: MotifIdx) -> Option<bool>;

    /// Returns an option with `true` if the motif under `index` is a mark. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_mark(&self, index: MotifIdx) -> Option<bool>;

    /// Returns true if `source` and `target` motifs are connected in _any_ direction. This is the
    /// most forgiving connection comparator - use `are_connected` to check whether two motifs are
    /// connected in a certain direction, or `bi_connected` to check whether two motifs are connected
    /// in both directions.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  -------------------------------------------------
    ///       [a] >----> [b]             ambi(a, b) = true
    ///  -------------------------------------------------
    ///       [a] >----> [b]             ambi(b, a) = true
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert!(weave.are_ambi_connected(a, b));
    /// assert!(weave.are_ambi_connected(b, a));
    /// ```
    fn are_ambi_connected(&self, source: MotifIdx, target: MotifIdx) -> bool;

    /// Returns true if `source` and `target` motifs are connected in _both_ directions. This is the
    /// least forgiving connection comparator - use `are_ambi_connected` to check whether two motifs
    /// are connected in any direction, or `connected` to check whether two motifs are connected
    /// in a certain direction (source to target).
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ------------------------------------------------
    ///       [a] >----> [b]             bi(a, b) = false
    ///  ------------------------------------------------
    ///       [a] >----> [b]             bi(b, a) = false
    ///  ------------------------------------------------
    ///       [a] >----> [b]             bi(a, b) = true
    ///       [a] <----< [b]
    ///  ------------------------------------------------
    ///       [a] >----> [b]             bi(b, a) = true
    ///       [a] <----< [b]
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert!(!weave.are_bi_connected(a, b));
    /// assert!(!weave.are_bi_connected(b, a));
    /// weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// assert!(weave.are_bi_connected(a, b));
    /// assert!(weave.are_bi_connected(b, a));
    /// ```
    fn are_bi_connected(&self, source: MotifIdx, target: MotifIdx) -> bool;

    /// Returns true if `source` and `target` motifs are connected in _the_ direction specified by
    /// the `source -> target` flow. Use `are_ambi_connected` to check whether two motifs are
    /// connected in any direction, or `are_bi_connected` to check whether two motifs are connected
    /// both directions.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  --------------------------------------------------
    ///       [a] >----> [b]             conn(a, b) = true
    ///  --------------------------------------------------
    ///       [a] >----> [b]             conn(b, a) = false
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert!(weave.are_connected(a, b));
    /// assert!(!weave.are_connected(b, a));
    /// ```
    fn are_connected(&self, source: MotifIdx, target: MotifIdx) -> bool;

    /// Returns the in degree of the `index` motif, or the number of arrows going into the motif.
    /// If index is malformed or doesn't exist, the function returns `None`.
    fn get_in_degree(&self, index: MotifIdx) -> Option<usize>;

    /// Returns the out degree of the `index` motif, or the number of arrows going out from the motif.
    /// If index is malformed or doesn't exist, the function returns `None`.
    fn get_out_degree(&self, index: MotifIdx) -> Option<usize>;

    /// Returns the total degree of the `index` motif, or the number of arrows going into or out
    /// from the motif. If index is malformed or doesn't exist, the function returns `None`.
    fn get_in_out_degree(&self, index: MotifIdx) -> Option<usize>;

    /// Returns the loop degree of the `index` motif, or the number of arrows going from the motif
    /// back into itself. If index is malformed or doesn't exist, the function returns `None`.
    fn get_loop_degree(&self, index: MotifIdx) -> Option<usize>;

    /// Returns a vector of indices of all the arrows between `source` and `target`. If either
    /// `source` or `target` are malformed, the result is an empty container. This function respects
    /// the `source -> target` flow order when returning connections.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ------------------------------------------------------
    ///       [a] >--c--> [b]            conns(a, b) = { c }
    ///       [a] <--d--< [b]
    ///  ------------------------------------------------------
    ///       [a] >--c--> [b]            conns(a, b) = { c, e }
    ///       [a] <--d--< [b]
    ///       [a] >--e--> [b]
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// let d = weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_arrows(a, b), vec![ c ]);
    /// let e = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_arrows(a, b).into_iter().sorted().collect::<Vec<usize>>(), vec![ c, e ]);
    /// ```
    fn get_arrows(&self, source: MotifIdx, target: MotifIdx) -> Vec<MotifIdx>;

    /// Gets the loop connections between `index` and itself. Equivalent to:
    /// `weave.get_arrows(index, index)`.
    fn get_loop_arrows(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Returns a vector of indices of all the arrows with `source` as source. If `source` is
    /// malformed, the result is an empty container. This function respects the `source -> target`
    /// flow order when returning connections.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ------------------------------------------------------
    ///       [a] >--c--> [b]          conns_from(a) = { c, e }
    ///       [a] <--d--< [b]
    ///       [a] >--e--> [b]
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// let d = weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// let e = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_arrows_from(a).into_iter().sorted().collect::<Vec<usize>>(), vec![ c, e ]);
    /// ```
    fn get_arrows_from(&self, source: MotifIdx) -> Vec<MotifIdx>;

    /// Returns a vector of indices of all the arrows with `target` as target. If `source` is
    /// malformed, the result is an empty container. This function respects the `source -> target`
    /// flow order when returning connections.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ------------------------------------------------------
    ///       [a] >--c--> [b]          conns_to(b) = { c, e }
    ///       [a] <--d--< [b]
    ///       [a] >--e--> [b]
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// let d = weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// let e = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_arrows_to(b).into_iter().sorted().collect::<Vec<usize>>(), vec![ c, e ]);
    /// ```
    fn get_arrows_to(&self, target: MotifIdx) -> Vec<MotifIdx>;

    /// Returns a vector of indices of all the arrows with `index` as a source or a target. If
    /// `source` is malformed, the result is an empty container. This function respects the
    /// `source -> target` flow order when returning connections. It is exactly the same as getting
    /// `get_arrows_from` and `get_arrows_to` on the same `index`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ------------------------------------------------------
    ///       [a] >--c--> [b]          conns_to(b) = { c, d, e }
    ///       [a] <--d--< [b]
    ///       [a] >--e--> [b]
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// let d = weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// let e = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_ambi_arrows(b).into_iter().sorted().collect::<Vec<usize>>(), vec![ c, d, e ]);
    /// ```
    fn get_ambi_arrows(&self, index: MotifIdx) -> Vec<MotifIdx> {
        let mut result: HashSet<MotifIdx> = HashSet::from_iter(self.get_arrows_from(index));
        result.extend(self.get_arrows_to(index));
        result.into_iter().collect()
    }

    /// Gets the immediate neighbors of this motif. For the purposes of this function, a neighbor
    /// is any entity that is at the end of an **arrow** from another motif.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ----------------------------------------------------
    ///       [a] >----> [b]          neighbors(a) = { b, c }
    ///       [a] >----> [c]
    ///       [a] <----< [d]
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_knot();
    /// let d = weave.new_knot();
    /// weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// weave.new_arrow(a, c).unwrap_or(weave.bottom());
    /// weave.new_arrow(d, a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_neighbors(a).into_iter().sorted().collect::<Vec<usize>>(), vec![ b, c ]);
    /// ```
    fn get_neighbors(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets the immediate co-neighbors of this motif. If `index` is malformed, the result is an
    /// empty container. For the purposes of this function, a co-neighbor is the dual of a neighbor:
    /// if one motif is a neighbor to another, then the other is a co-neighbor to the first.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ------------------------------------------------------
    ///       [a] >----> [b]         co_neighbors(b) = { a, c }
    ///       [c] >----> [b]
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_knot();
    /// weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// weave.new_arrow(c, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_co_neighbors(b), vec![ a, c ]);
    /// ```
    fn get_co_neighbors(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets the immediate neighbors **and** co-neighbors of this motif. If `index` is malformed,
    /// the result is an empty container. The union of all neighbors and co-neighbors means that
    /// this function returns **all** the motifs that are connected to and from the `index` motif
    /// via an arrow.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  ------------------------------------------------------
    ///   [a]>---->[b]>--->[c]     ambi_neighbors(b) = { a, c }
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_knot();
    /// weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// weave.new_arrow(b, c).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_ambi_neighbors(b), vec![ a, c ]);
    /// ```
    fn get_ambi_neighbors(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets all the tethers of the motif under `index`. If `index` is malformed, the result is an
    /// empty container.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  --------------------------------------------------
    ///       [a] ----->(b)           tethers(a) = { b, c }
    ///       [a] ----->(c)
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_tether(a).unwrap_or(weave.bottom());
    /// let c = weave.new_tether(a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_tethers(a).into_iter().sorted().collect::<Vec<usize>>(), vec![ b, c ]);
    /// ```
    fn get_tethers(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets all the marks of the motif under `index`. If `index` is malformed, the result is an
    /// empty container.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  --------------------------------------------------
    ///       [a] <-----(b)             marks(a) = { b, c }
    ///       [a] <-----(c)
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_mark(a).unwrap_or(weave.bottom());
    /// let c = weave.new_mark(a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_marks(a).into_iter().sorted().collect::<Vec<usize>>(), vec![ b, c ]);
    /// ```
    fn get_marks(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets all the hoisted arrows between `source_index` and `target_index`. A *hoisted arrow* is
    /// a specific construction in the form of `[a] --->(b) >---c---> (d)---> [e]`, comprising a
    /// tether, arrow, and mark. If this construction is found, `c` is referred to as being hoisted
    /// between `[a]` and `[e]` by the meta-motifs `(b)` and `(d)`. Hoisted arrows are useful to
    /// represent arrows in a meta structure, that isn't a part of some graph.
    fn get_hoisted_arrows(&self, source_index: MotifIdx, target_index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets all the hoisted arrows going from the `index` motif. A *hoisted arrow* is a specific
    /// construction in the form of `[a] --->(b) >---c---> (d)---> [e]`, comprising a tether, arrow,
    /// and mark. If this construction is found, `c` is referred to as being hoisted between `[a]`
    /// and `[e]` by the meta-motifs `(b)` and `(d)`. Hoisted arrows are useful to represent arrows
    /// in a meta structure, that isn't a part of some graph (for example, hierarchies).
    fn get_hoisted_arrows_from(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets all the hoisted arrows going to the `index` motif. A *hoisted arrow* is a specific
    /// construction in the form of `[a] --->(b) >---c---> (d)---> [e]`, comprising a tether, arrow,
    /// and mark. If this construction is found, `c` is referred to as being hoisted between `[a]`
    /// and `[e]` by the meta-motifs `(b)` and `(d)`.
    fn get_hoisted_arrows_to(&self, index: MotifIdx) -> Vec<MotifIdx>;

    /// Gets the hoist endpoints of a hoisted arrow specified by `index`. A *hoisted arrow* is a
    /// specific construction in the form of `[a] --->(b) >---c---> (d)---> [e]`, comprising a
    /// tether, arrow, and mark. If this construction is found, `c` is referred to as being hoisted
    /// between `[a]` and `[e]` by the meta-motifs `(b)` and `(d)`. The endpoints of a hoisted arrow
    /// are exactly the two motifs - `[a]` and `[e]`. Returns `None` if `index` does not specify a
    /// hoisted arrow, or is otherwise malformed.
    fn get_hoist_endpoints(&self, index: MotifIdx) -> Option<(MotifIdx, MotifIdx)>;

    /// Given a knot `index`, gets the flow cover of this graph: the set of all knots that are
    /// connected with `index` via arrows. The cover recognizes `source` to `target` arrow flow, and
    /// counts only the knots that are _connected_ in flow-order. The cover will change depending on
    /// the representative chosen only to include forward-facing nodes up to transitive closure. If
    /// the `index` motif is not a knot, the result will simply be an empty cover.
    ///
    /// Note: the `get_flow_graph_cover` function does *not* use a recursive solution instead building up
    /// a queue of next values to visit and collecting nodes in breadth-first fashion.
    ///
    /// Note: due to the hashing operation, the cover's `knots` vector will be ordered by `index`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                            post
    ///  ----------------------------------------------------------------------
    ///   [d]<---<[a]<---<[b]>--->[c]            flow_cover(a) = { a, d }
    ///  ----------------------------------------------------------------------
    ///   [d]<---<[a]<---<[b]>--->[c]            flow_cover(b) = { a, b, c, d }
    ///  ----------------------------------------------------------------------
    ///   [d]<---<[a]<---<[b]>--->[c]            flow_cover(c) = { c }
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_knot();
    /// let d = weave.new_knot();
    /// let _ = weave.new_arrow(b, a);
    /// let _ = weave.new_arrow(a, d);
    /// let _ = weave.new_arrow(b, c);
    /// let cover = weave.get_flow_graph_cover(a).knots;
    /// assert_eq!(cover.into_iter().sorted().collect::<Vec<usize>>(), vec![ a, d ]);
    /// let cover = weave.get_flow_graph_cover(b).knots;
    /// assert_eq!(cover.into_iter().sorted().collect::<Vec<usize>>(), vec![ a, b, c, d ]);
    /// let cover = weave.get_flow_graph_cover(c).knots;
    /// assert_eq!(cover.into_iter().sorted().collect::<Vec<usize>>(), vec![ c ]);
    /// ```
    fn get_flow_graph_cover(&self, knot_index: MotifIdx) -> Cover;

    /// Given a knot `index`, gets the cover of this graph: the set of all knots that are connected
    /// with `index` via arrows. The cover doesn't recognize `source` to `target` arrow flow, and
    /// counts all the knots as if they are _ambi-connected_. The cover will be the same regardless
    /// of representative `index` chosen.  If the `index` motif is not a knot, the result will
    /// simply be an empty cover.
    ///
    /// Note: the `get_graph_cover` function does *not* use a recursive solution instead building up
    /// a queue of next values to visit and collecting nodes in breadth-first fashion.
    ///
    /// Note: due to the hashing operation, the cover's `knots` vector will be ordered by `index`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                                     post
    ///  -----------------------------------------------------------------
    ///   [d]<---<[a]<---<[b]>--->[c]            cover(a) = { a, b, c, d }
    /// ```
    ///
    /// ```
    /// use itertools::Itertools;
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_knot();
    /// let d = weave.new_knot();
    /// let _ = weave.new_arrow(b, a);
    /// let _ = weave.new_arrow(a, d);
    /// let _ = weave.new_arrow(b, c);
    /// let cover = weave.get_graph_cover(a).knots;
    /// assert_eq!(cover.into_iter().sorted().collect::<Vec<usize>>(), vec![ a, b, c, d ]);
    /// ```
    fn get_graph_cover(&self, knot_index: MotifIdx) -> Cover;

    /// Finds all embeddings of one graph in another. The `embed_relation` needs to be a hoisted
    /// arrow between two arbitrary representative nodes from the graphs. The source of this relation
    /// needs to be the graph that we are looking to find, while the target is the graph we are
    /// searching in. The `Embedding` structure will contain the `embed_relation` index, as well as
    /// an `image` mapping between the motifs in the `source` and `target` graphs.
    ///
    /// Guarantees: the `embed_relation` connects two knots from two different graphs (checked by
    /// creating the two `Cover` graphs and comparing their hashes). If the graphs are the same,
    /// the result is `None`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///                pre                                      post
    ///  ---------------------------------------------------------------
    ///       (t)>-----embed----->(m)               find_embeddings(embed) = {
    ///       ^                    \                    { { a, c }, { b, d } },
    ///       |                     v                   { { a, c }, { b, e } },
    ///     [a]>-->[b]          /-<[c]>-\               { { a, d }, { b, e } }
    ///                         |       |           }
    ///                         v       v
    ///                        [d]>--->[e]
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// use libweave::embeddings::pattern_matching::PatternMatchingEmbedding;
    /// 
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let _ab = weave.new_arrow(a, b).unwrap();
    ///
    /// // data
    /// let c = weave.new_knot();
    /// let d = weave.new_knot();
    /// let e = weave.new_knot();
    /// let _cd = weave.new_arrow(c, d).unwrap();
    /// let _ce = weave.new_arrow(c, e).unwrap();
    /// let _de = weave.new_arrow(d, e).unwrap();
    ///
    /// // hoist
    /// let t = weave.new_tether(a).unwrap();
    /// let m = weave.new_mark(c).unwrap();
    /// let embed = weave.new_arrow(t, m).unwrap();
    ///
    /// let matches = weave.find_all_embeddings::<PatternMatchingEmbedding>(embed);
    /// assert!(matches.is_some());
    /// let embeddings = matches.unwrap();
    /// assert_eq!(embeddings.len(), 3);
    /// assert_eq!(embeddings[0].image, vec![ (a, c), (b, d) ]);
    /// assert_eq!(embeddings[1].image, vec![ (a, c), (b, e) ]);
    /// assert_eq!(embeddings[2].image, vec![ (a, d), (b, e) ]);
    /// ```
    fn find_all_embeddings<FE: FindEmbeddings>(&self, embed_relation: MotifIdx) -> Option<Vec<Embedding>>;

    /// Finds one embedding of one graph in another. The `embed_relation` needs to be a hoisted
    /// arrow between two arbitrary representative nodes from the graphs. The source of this relation
    /// needs to be the graph that we are looking to find, while the target is the graph we are
    /// searching in. The `Embedding` structure will contain the `embed_relation` index, as well as
    /// an `image` mapping between the motifs in the `source` and `target` graphs.
    ///
    /// Guarantees: the `embed_relation` connects two knots from two different graphs (checked by
    /// creating the two `Cover` graphs and comparing their hashes). If the graphs are the same,
    /// the result is `None`.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///                pre                                      post
    ///  ---------------------------------------------------------------
    ///       (t)>-----embed----->(m)               find_one_embedding(embed) = {
    ///       ^                    \                    { a, c }, { b, d },
    ///       |                     v               }
    ///     [a]>-->[b]          /-<[c]>-\
    ///                         |       |
    ///                         v       v
    ///                        [d]>--->[e]
    /// ```
    ///
    /// ```
    /// use libweave::weave::{Weave, Weaveable};
    /// use libweave::embeddings::pattern_matching::PatternMatchingEmbedding;
    ///
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let _ab = weave.new_arrow(a, b).unwrap();
    ///
    /// // data
    /// let c = weave.new_knot();
    /// let d = weave.new_knot();
    /// let e = weave.new_knot();
    /// let _cd = weave.new_arrow(c, d).unwrap();
    /// let _ce = weave.new_arrow(c, e).unwrap();
    /// let _de = weave.new_arrow(d, e).unwrap();
    ///
    /// // hoist
    /// let t = weave.new_tether(a).unwrap();
    /// let m = weave.new_mark(c).unwrap();
    /// let embed = weave.new_arrow(t, m).unwrap();
    ///
    /// let matches = weave.find_one_embedding::<PatternMatchingEmbedding>(embed);
    /// assert!(matches.is_some());
    /// let embeddings = matches.unwrap();
    /// assert_eq!(embeddings.image, vec![ (a, c), (b, d) ]);
    /// ```
    fn find_one_embedding<FE: FindEmbeddings>(&self, embed_relation: MotifIdx) -> Option<Embedding>;
}

#[repr(C)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
#[derive(Clone, Debug)]
pub struct WeaveRef<'s>(pub(crate) Box<RefCell<WeaveInternal<'s>>>, pub(crate) MotifIdx);

pub type Weave<'w, 's> = &'w WeaveRef<'s>;

impl<'s> WeaveRef<'s> {
    pub fn bottom(&self) -> MotifIdx { self.1 }
}

// Type aliases are removed in impl for easier translation into ffi

fn weave_alloc(internal: &mut RefMut<WeaveInternal>, motif: Motif) -> MotifId {
    if let Some(free) = internal.motif_freelist.pop_front() {
        *internal.motif_space.get_mut(free).unwrap() = motif;
        free
    } else {
        internal.motif_space.alloc(motif)
    }
}

impl<'w, 's> Weaveable<WeaveRef<'s>> for Weave<'w, 's> {
    fn create() -> WeaveRef<'s> {
        let wv = Box::new(RefCell::new(WeaveInternal::default()));
        let bt = wv.as_ref().borrow().motif_bottom;
        WeaveRef(wv, bt)
    }

    fn get_source(&self, index: usize) -> usize {
        let internal = self.0.borrow_mut();
        if let Some(motif_id) = internal.motif_index.get(&index) {
            if let Some(motif) = internal.motif_space.get(*motif_id) {
                return match motif {
                    Motif::Knot => index,
                    Motif::Arrow { source, .. } => source.index(),
                    Motif::Tether { source } => source.index(),
                    Motif::Mark { .. } => index,
                }
            }
        }

        self.bottom()
    }

    fn get_source_nth(&self, index: usize, degree: usize) -> usize {
        if degree == 0 { return index }

        if degree == 1 {
            self.get_source(index)
        } else {
            self.get_source_nth(self.get_source(index), degree - 1)
        }
    }

    fn get_target(&self, index: usize) -> usize {
        let internal = self.0.borrow_mut();
        if let Some(motif_id) = internal.motif_index.get(&index) {
            if let Some(motif) = internal.motif_space.get(*motif_id) {
                return match motif {
                    Motif::Knot => index,
                    Motif::Arrow { target, .. } => target.index(),
                    Motif::Tether { .. } => index,
                    Motif::Mark { target } => target.index(),
                }
            }
        }

        self.bottom()
    }

    fn get_target_nth(&self, index: usize, degree: usize) -> usize {
        if degree == 0 { return index }

        if degree == 1 {
            self.get_target(index)
        } else {
            self.get_target_nth(self.get_target(index), degree - 1)
        }
    }

    fn new_knot(&self) -> usize {
        let mut internal = self.0.borrow_mut();
        let knot = weave_alloc(&mut internal, Motif::Knot);
        let index = knot.index();
        internal.motif_index.insert(index, knot);
        index
    }

    fn new_arrow(&self, source_index: usize, target_index: usize) -> Option<usize> {
        let mut internal = self.0.borrow_mut();
        let bottom = self.bottom();
        if source_index == bottom { return None; }
        if target_index == bottom { return None; }

        let source_valid = internal.motif_index.get(&source_index).copied();
        let target_valid = internal.motif_index.get(&target_index).copied();
        if let (Some(source), Some(target)) = (source_valid, target_valid) {
            let arrow = weave_alloc(&mut internal, Motif::Arrow { source, target });
            let index = arrow.index();
            internal.motif_index.insert(index, arrow);
            internal.motif_conns.insert((source_index, target_index), index);
            internal.motif_neighbors.insert(source_index, target_index);
            internal.motif_co_neighbors.insert(target_index, source_index);
            Some(index)
        } else {
            None
        }
    }

    fn new_tether(&self, source_index: usize) -> Option<usize> {
        let mut internal = self.0.borrow_mut();
        let bottom = self.bottom();
        if source_index == bottom { return None; }

        if let Some(source) = internal.motif_index.get(&source_index).copied() {
            let tether = weave_alloc(&mut internal, Motif::Tether { source });
            let index = tether.index();
            internal.motif_index.insert(index, tether);
            internal.motif_tethers.insert(source_index, index);
            Some(index)
        } else {
            None
        }
    }

    fn new_mark(&self, target_index: usize) -> Option<usize> {
        let mut internal = self.0.borrow_mut();
        let bottom = self.bottom();
        if target_index == bottom { return None; }

        if let Some(target) = internal.motif_index.get(&target_index).copied() {
            let mark = weave_alloc(&mut internal, Motif::Mark { target });
            let index = mark.index();
            internal.motif_index.insert(index, mark);
            internal.motif_marks.insert(target_index, index);
            Some(index)
        } else {
            None
        }
    }

    fn delete(&self, target: usize) -> bool {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut delete_set = HashSet::new();

        queue.push_back(target);
        delete_set.insert(target);

        while let Some(motif) = queue.pop_front() {
            if !visited.contains(&motif) {
                visited.insert(motif);

                if !self.is_knot(motif).unwrap_or(true) {
                    delete_set.insert(motif);
                }

                queue.extend(self.get_arrows_to(motif));
                queue.extend(self.get_arrows_from(motif));
                queue.extend(self.get_marks(motif));
                queue.extend(self.get_tethers(motif));
            }
        }

        let mut internal = self.0.borrow_mut();
        for del in &delete_set {
            if let Some(&id) = internal.motif_index.get(del) {
                let remove_arrow_endpoints = internal.motif_conns.iter().filter(|&c| {
                    let (a, b) = c;
                    &a.0 == del || &a.1 == del || b == del
                }).map(|c| *c.0).collect::<Vec<_>>();

                for rem in remove_arrow_endpoints {
                    internal.motif_conns.remove(&rem);
                }

                let remove_from_neighbors = internal.motif_neighbors.get_vec(del).unwrap_or(&vec![]).to_vec();
                internal.motif_co_neighbors.remove(del);
                for rem in remove_from_neighbors {
                    internal.motif_neighbors.remove(&rem);
                }

                let remove_from_co_neighbors = internal.motif_neighbors.get_vec(del).unwrap_or(&vec![]).to_vec();
                internal.motif_neighbors.remove(del);
                for rem in remove_from_co_neighbors {
                    internal.motif_co_neighbors.remove(&rem);
                }

                internal.motif_marks.remove(del);
                internal.motif_tethers.remove(del);
                internal.motif_freelist.push_back(id);
                internal.motif_index.remove(del);
            }
        }

        !delete_set.is_empty()
    }

    fn exists(&self, index: usize) -> bool {
        let internal = self.0.borrow();
        internal.motif_index.contains_key(&index)
    }

    fn identify(&self, index: usize) -> MotifId {
        let internal = self.0.borrow();
        if let Some(motif) = internal.motif_index.get(&index) {
            *motif
        } else {
            internal.motif_index.get(&self.bottom()).copied().unwrap()
        }
    }

    fn opt_identify(&self, index: usize) -> Option<MotifId> {
        let internal = self.0.borrow();
        internal.motif_index.get(&index).copied()
    }

    fn is_knot(&self, index: usize) -> Option<bool> {
        let internal = self.0.borrow_mut();
        if let Some(id) = internal.motif_index.get(&index).copied() {
            if let Some(e) = internal.motif_space.get(id).copied() {
                return Some(matches!(e, Motif::Knot));
            }
        }

        None
    }

    fn is_arrow(&self, index: usize) -> Option<bool> {
        let internal = self.0.borrow_mut();
        if let Some(id) = internal.motif_index.get(&index).copied() {
            if let Some(e) = internal.motif_space.get(id).copied() {
                return Some(matches!(e, Motif::Arrow { .. }));
            }
        }

        None
    }

    fn is_tether(&self, index: usize) -> Option<bool> {
        let internal = self.0.borrow_mut();
        if let Some(id) = internal.motif_index.get(&index).copied() {
            if let Some(e) = internal.motif_space.get(id).copied() {
                return Some(matches!(e, Motif::Tether { .. }));
            }
        }

        None
    }

    fn is_mark(&self, index: usize) -> Option<bool> {
        let internal = self.0.borrow_mut();
        if let Some(id) = internal.motif_index.get(&index).copied() {
            if let Some(e) = internal.motif_space.get(id).copied() {
                return Some(matches!(e, Motif::Mark { .. }));
            }
        }

        None
    }

    fn are_ambi_connected(&self, source_index: usize, target_index: usize) -> bool {
        let internal = self.0.borrow_mut();
        let bottom = self.bottom();
        if source_index == bottom || target_index == bottom { return false; }

        internal.motif_conns.contains_key(&(source_index, target_index))
            || internal.motif_conns.contains_key(&(target_index, source_index))
    }

    fn are_bi_connected(&self, source_index: usize, target_index: usize) -> bool {
        let internal = self.0.borrow_mut();
        let bottom = self.bottom();
        if source_index == bottom || target_index == bottom { return false; }

        internal.motif_conns.contains_key(&(source_index, target_index))
            && internal.motif_conns.contains_key(&(target_index, source_index))
    }

    fn are_connected(&self, source_index: usize, target_index: usize) -> bool {
        let internal = self.0.borrow_mut();
        let bottom = self.bottom();
        if source_index == bottom || target_index == bottom { return false; }

        internal.motif_conns.contains_key(&(source_index, target_index))
    }

    fn get_in_degree(&self, index: usize) -> Option<usize> {
        let internal = self.0.borrow();
        let mut result = 0;

        if !internal.motif_index.contains_key(&index) {
            return None;
        }

        if internal.motif_co_neighbors.contains_key(&index) {
            for neighbor in internal.motif_co_neighbors.get_vec(&index).unwrap() {
                result += internal.motif_conns.get_vec(&(*neighbor, index)).unwrap().len();
            }

            return Some(result);
        }

        Some(0)
    }

    fn get_out_degree(&self, index: usize) -> Option<usize> {
        let internal = self.0.borrow();
        let mut result = 0;

        if !internal.motif_index.contains_key(&index) {
            return None;
        }

        if internal.motif_neighbors.contains_key(&index) {
            for neighbor in internal.motif_neighbors.get_vec(&index).unwrap() {
                result += internal.motif_conns.get_vec(&(index, *neighbor)).unwrap().len();
            }

            return Some(result);
        }

        Some(0)
    }

    fn get_in_out_degree(&self, index: usize) -> Option<usize> {
        if let (Some(in_d), Some(out_d)) = (self.get_in_degree(index), self.get_out_degree(index)) {
            Some(in_d + out_d)
        } else {
            None
        }
    }

    fn get_loop_degree(&self, index: usize) -> Option<usize> {
        Some(self.get_arrows(index, index).len())
    }

    fn get_arrows(&self, source: usize, target: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_conns.get_vec(&(source, target)).cloned().unwrap_or(vec![]).to_vec()
    }

    fn get_loop_arrows(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_conns.get_vec(&(index, index)).cloned().unwrap_or(vec![]).to_vec()
    }

    fn get_arrows_from(&self, source: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        let mut results = HashSet::new();
        if internal.motif_neighbors.contains_key(&source) {
            for neighbor in internal.motif_neighbors.get_vec(&source).unwrap() {
                for a in internal.motif_conns.get_vec(&(source, *neighbor)).unwrap() {
                    results.insert(*a);
                }
            }
        }

        results.into_iter().collect()
    }

    fn get_arrows_to(&self, target: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        let mut results = HashSet::new();
        if internal.motif_co_neighbors.contains_key(&target) {
            for neighbor in internal.motif_co_neighbors.get_vec(&target).unwrap() {
                for a in internal.motif_conns.get_vec(&(*neighbor, target)).unwrap() {
                    results.insert(*a);
                }
            }
        }

        results.into_iter().collect()
    }

    fn get_neighbors(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_neighbors.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_co_neighbors(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_co_neighbors.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_ambi_neighbors(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        let mut set: HashSet<usize> = HashSet::from_iter(
            internal.motif_neighbors.get_vec(&index).unwrap_or(&vec![]).to_vec());
        set.extend(internal.motif_co_neighbors.get_vec(&index).unwrap_or(&vec![]).to_vec());
        set.into_iter().sorted().collect()
    }

    fn get_tethers(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_tethers.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_marks(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_marks.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_hoisted_arrows(&self, source_index: usize, target_index: usize) -> Vec<usize> {
        self.get_tethers(source_index).iter()
            .flat_map(|t| self.get_arrows_from(*t))
            .filter(|a| {
                let tgt = self.get_target(*a);
                self.is_mark(tgt).unwrap_or(false) && self.get_target(tgt) == target_index
            })
            .collect()
    }

    fn get_hoisted_arrows_from(&self, index: usize) -> Vec<usize> {
        self.get_tethers(index).iter()
            .flat_map(|t| self.get_arrows_from(*t))
            .filter(|a| self.is_mark(self.get_target(*a)).unwrap_or(false))
            .collect()
    }

    fn get_hoisted_arrows_to(&self, index: usize) -> Vec<usize> {
        self.get_marks(index).iter()
            .flat_map(|t| self.get_arrows_to(*t))
            .filter(|a| self.is_tether(self.get_source(*a)).unwrap_or(false))
            .collect()
    }

    fn get_hoist_endpoints(&self, index: usize) -> Option<(usize, usize)> {
        if self.is_arrow(index).unwrap_or(false) {
            let (tether, mark) = (self.get_source(index), self.get_target(index));

            if self.is_tether(tether).unwrap_or(false)
                && self.is_mark(mark).unwrap_or(false) {
                return Some((self.get_source(tether), self.get_target(mark)));
            }
        }

        None
    }

    fn get_flow_graph_cover(&self, knot_index: usize) -> Cover {
        let reserved = { self.0.borrow().motif_index.len() * 2 };
        let mut visited = HashSet::with_capacity(reserved);
        let mut queue = VecDeque::with_capacity(reserved);

        queue.push_back(knot_index);

        while let Some(next) = queue.pop_front() {
            if !visited.contains(&next) {
                visited.insert(next);

                let neighbors = self.get_neighbors(next);
                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        Cover::from(visited.iter().sorted().cloned().collect::<Vec<usize>>())
    }

    fn get_graph_cover(&self, knot_index: usize) -> Cover {
        let reserved = { self.0.borrow().motif_index.len() * 2 };
        let mut visited = HashSet::with_capacity(reserved);
        let mut queue = VecDeque::with_capacity(reserved);

        queue.push_back(knot_index);

        while let Some(next) = queue.pop_front() {
            if !visited.contains(&next) {
                visited.insert(next);

                let neighbors = self.get_neighbors(next);
                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }

                let co_neighbors = self.get_co_neighbors(next);
                for co_neighbor in co_neighbors {
                    if !visited.contains(&co_neighbor) {
                        queue.push_back(co_neighbor);
                    }
                }
            }
        }

        Cover::from(visited.iter().sorted().cloned().collect::<Vec<usize>>())
    }

    fn find_all_embeddings<FE: FindEmbeddings>(&self, embed_relation: usize) -> Option<Vec<Embedding>> {
        if let Some((query_repr_index, data_repr_index))
            = self.get_hoist_endpoints(embed_relation) {
            let query_graph = self.get_graph_cover(query_repr_index);
            let data_graph = self.get_graph_cover(data_repr_index);
            if query_graph.hash == data_graph.hash { return None; }

            return Some(FE::find_all_embeddings(self, embed_relation, query_graph, data_graph));
        }

        None
    }

    fn find_one_embedding<FE: FindEmbeddings>(&self, embed_relation: MotifIdx) -> Option<Embedding> {
        if let Some((query_repr_index, data_repr_index))
            = self.get_hoist_endpoints(embed_relation) {
            let query_graph = self.get_graph_cover(query_repr_index);
            let data_graph = self.get_graph_cover(data_repr_index);
            if query_graph.hash == data_graph.hash { return None; }

            return FE::find_one_embedding(self, embed_relation, query_graph, data_graph);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::embeddings::pattern_matching::PatternMatchingEmbedding;
    use crate::weave::{Weave, Weaveable};

    #[test]
    fn test_alloc_motif_is_not_bottom() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        assert_ne!(weave.bottom(), a);

        let b = weave.new_arrow(a, a).unwrap();
        assert_ne!(weave.bottom(), b);

        let c = weave.new_tether(a).unwrap();
        assert_ne!(weave.bottom(), c);

        let d = weave.new_mark(a).unwrap();
        assert_ne!(weave.bottom(), d);
    }

    /*
        [a]>--->[b]

        1. [a] has a neighbor in [b]
        2. [b] has no neighbor (as there's no arrow going to it)
    */
    #[test]
    fn test_get_neighbors() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let ab = weave.new_arrow(a, b);

        assert!(ab.is_some());

        assert!(weave.are_connected(a, b));
        assert_eq!(weave.get_neighbors(a), vec![ b ]);

        assert!(!weave.are_connected(b, a));
        assert_eq!(weave.get_neighbors(b), vec![]);
    }

    /*
        [a]>--->[b]

        1. [b] has a co-neighbor in [a] (because [a] has a neighbor in [b])
        2. [a] has no co-neighbors (as there's no arrow going into it)
    */
    #[test]
    fn test_get_co_neighbors() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let ab = weave.new_arrow(a, b);

        assert!(ab.is_some());

        assert!(weave.are_connected(a, b));
        assert_eq!(weave.get_co_neighbors(a), vec![]);

        assert!(!weave.are_connected(b, a));
        assert_eq!(weave.get_co_neighbors(b), vec![ a ]);
    }

    #[test]
    fn test_get_ambi_neighbors() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let c = weave.new_knot();
        let _ = weave.new_arrow(a, b).unwrap();
        let _ = weave.new_arrow(b, c).unwrap();

        assert_eq!(weave.get_ambi_neighbors(b).into_iter().sorted().collect::<Vec<usize>>(), vec![ a, c ]);
    }

    /*
        Ambi-connected: connected with at least one arrow in any direction
        Bi-connected: connected with at least one arrow in both directions

        ambi(x, y) == ambi(x, y)  -and-   bi(x, y) == bi(y, x)

        a)
            [a]>-c-->[b]

        b)  [a]>-c-->[b]
             ^---d----v
     */
    #[test]
    fn test_bi_connectivity() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let _c = weave.new_arrow(a, b);

        assert!(weave.are_ambi_connected(a, b));
        assert!(weave.are_ambi_connected(b, a));

        assert!(!weave.are_bi_connected(a, b));
        assert!(!weave.are_bi_connected(b, a));

        let _d = weave.new_arrow(b, a);

        assert!(weave.are_ambi_connected(a, b));
        assert!(weave.are_ambi_connected(b, a));

        assert!(weave.are_bi_connected(a, b));
        assert!(weave.are_bi_connected(b, a));
    }

    /*
        [a]>---ab1--->[b]
         v      |      ^
         |     phi     |
         |      |      |
         |      v      |
         \-----ab2-----/
    */
    #[test]
    fn test_arrows_can_have_neighbors() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let ab1 = weave.new_arrow(a, b).unwrap();
        let ab2 = weave.new_arrow(a, b).unwrap();
        assert_ne!(ab1, ab2);

        let _phi = weave.new_arrow(ab1, ab2).unwrap();
        assert!(weave.are_connected(ab1, ab2));
        assert_eq!(weave.get_neighbors(ab1), vec![ ab2 ]);
        assert_eq!(weave.get_neighbors(ab2), vec![]);

        assert_eq!(weave.get_co_neighbors(ab1), vec![]);
        assert_eq!(weave.get_co_neighbors(ab2), vec![ ab1 ]);
    }

    #[test]
    fn test_nth_degree_of_knot_is_knot() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        assert_eq!(weave.get_source(a), a);
        assert_eq!(weave.get_source_nth(a, 2), a);
        assert_eq!(weave.get_source_nth(a, 3), a);

        assert_eq!(weave.get_target(a), a);
        assert_eq!(weave.get_target_nth(a, 2), a);
        assert_eq!(weave.get_target_nth(a, 3), a);
    }

    /*
                ---
                \ /
        [a]>-f->[b]>-...
             |   ^
             |   |
         target--/

        target(f) = b, but
        target(target(f)) = target(b) = b
    */
    #[test]
    fn test_nth_degree_of_arrow_is_its_endpoint() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let f = weave.new_arrow(a, b).unwrap();

        assert_eq!(weave.get_target(f), b);
        assert_eq!(weave.get_target_nth(f, 2), b);
        assert_eq!(weave.get_target_nth(f, 3), b);
    }

    /*
        [a]--->(c)>--e-->(d)---->[b]
    */
    #[test]
    fn test_nth_degree_traversal_of_arrow_tether_and_marks() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let c = weave.new_tether(a).unwrap();
        let d = weave.new_mark(b).unwrap();
        let e = weave.new_arrow(c, d).unwrap();

        assert_eq!(weave.get_source(e), c);
        assert_eq!(weave.get_target(e), d);

        assert_eq!(weave.get_source(weave.get_source(e)), a);
        assert_eq!(weave.get_target(weave.get_target(e)), b);

        assert_eq!(weave.get_source_nth(e, 2), a);
        assert_eq!(weave.get_target_nth(e, 2), b);
    }

    /*
        [a]<-------(m)

        A mark targets another motif, but is itself not a knot.
        They are knot-like in the fact that they are endpoints.
        They are arrow-like in the fact that they "tie" to other motifs.
        Marks are good form for making properties of objects.
     */
    #[test]
    fn test_marks() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let m = weave.new_mark(a).unwrap();
        assert_ne!(m, weave.bottom());
        assert_eq!(weave.is_mark(m), Some(true));
        assert_eq!(weave.get_source(m), m);
        assert_eq!(weave.get_target(m), a);
        assert_eq!(weave.get_marks(a), vec![ m ]);
    }

    /*
        [a]------->(t)

        Other motifs tie to tethers, but tethers themselves aren't knots.
        They are knot-like in the fact that they are endpoints.
        They are arrow-like in the fact that other motifs "tie" to them.
        Tethers are good for representing hierarchies.
     */
    #[test]
    fn test_tethers() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let t = weave.new_tether(a).unwrap();
        assert_ne!(t, weave.bottom());
        assert_eq!(weave.is_tether(t), Some(true));
        assert_eq!(weave.get_source(t), a);
        assert_eq!(weave.get_target(t), t);
        assert_eq!(weave.get_tethers(a), vec![ t ]);
    }

    /*
        Hoisting an arrow between a tether and a mark makes it
        possible for us to add hierarchies to objects without
        bulking their arrow-sets with meta-arrows that aren't
        an *actual* part of a graph that we might want to traverse
        or search through.

                the parenthood relationship
                           |
        [child]-->(t)>--parent-->(m)-->[parent]
                   |              |
                   |         parent's mark to child
                   |
           child's tether to parent

       To find the child and parent, we can write the
       get_child_and_parent(parenthood) as follows:

           1.           (t)>--parent->(m)
                         |             |
                       source        target

           2. [child]-->(t)        (m)-->[parent]
                 |                           |
              source                       target

       Because we're working with marks and tethers, we can
       behave as if they're arrows, and get_source(get_source(p))
       becomes get_source_n(p, 2) and same for targets.
     */
    #[test]
    fn test_hoisting() {
        fn make_parent_hierarchy(weave: Weave, child: usize, parent: usize) -> Option<usize> {
            let t = weave.new_tether(child).unwrap();
            let m = weave.new_mark(parent).unwrap();
            weave.new_arrow(t, m)
        }

        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let parenthood = make_parent_hierarchy(weave, a, b).unwrap();
        let cp = weave.get_hoist_endpoints(parenthood);

        assert!(cp.is_some());
        let (child, parent) = cp.unwrap();
        assert_eq!(child, a);
        assert_eq!(parent, b);

        assert_eq!(weave.get_hoisted_arrows(a, b), vec![ parenthood ]);
        assert_eq!(weave.get_hoisted_arrows_from(a), vec![ parenthood ]);
        assert_eq!(weave.get_hoisted_arrows_to(b), vec![ parenthood ]);
    }

    #[test]
    fn test_ordering_in_unsorted_vector() {
        use itertools::Itertools;
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_mark(a).unwrap_or(weave.bottom());
        let c = weave.new_mark(a).unwrap_or(weave.bottom());
        assert_eq!(weave.get_marks(a).into_iter().sorted().collect::<Vec<usize>>(), vec![ b, c ]);
    }

    #[test]
    fn test_covers() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let c = weave.new_knot();
        let d = weave.new_knot();
        let _ = weave.new_arrow(b, a);
        let _ = weave.new_arrow(a, d);
        let _ = weave.new_arrow(b, c);
        let cover_a = weave.get_graph_cover(a);
        assert_eq!(cover_a.knots, vec![ a, b, c, d ]);
        let cover_b = weave.get_graph_cover(b);
        assert_eq!(cover_b.knots, vec![ a, b, c, d ]);
        assert_eq!(cover_a.hash, cover_b.hash);

        let flow_cover_a = weave.get_flow_graph_cover(a);
        assert_eq!(flow_cover_a.knots, vec![ a, d ]);
        let flow_cover_b = weave.get_flow_graph_cover(b);
        assert_eq!(flow_cover_b.knots, vec![ a, b, c, d ]);
        assert_ne!(flow_cover_a.hash, flow_cover_b.hash);
    }

    #[test]
    fn test_finding_embeddings() {
        let weave = &Weave::create();
        // query
        let a = weave.new_knot();
        let b = weave.new_knot();
        let _ab = weave.new_arrow(a, b).unwrap();

        // data
        let c = weave.new_knot();
        let d = weave.new_knot();
        let e = weave.new_knot();
        let _cd = weave.new_arrow(c, d).unwrap();
        let _ce = weave.new_arrow(c, e).unwrap();
        let _de = weave.new_arrow(d, e).unwrap();

        // hoist
        let t = weave.new_tether(a).unwrap();
        let m = weave.new_mark(c).unwrap();
        let embed = weave.new_arrow(t, m).unwrap();

        let matches = weave.find_all_embeddings::<PatternMatchingEmbedding>(embed);
        assert!(matches.is_some());
        let embeddings = matches.unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].image, vec![ (a, c), (b, d) ]);
        assert_eq!(embeddings[1].image, vec![ (a, c), (b, e) ]);
        assert_eq!(embeddings[2].image, vec![ (a, d), (b, e) ]);
    }

    #[test]
    fn test_delete_cascades_and_reuses_ids() {
        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_mark(a).unwrap_or(weave.bottom());
        assert!(weave.is_mark(b).unwrap());
        assert!(!weave.is_knot(b).unwrap());
        weave.delete(a);
        assert!(!weave.exists(a));
        assert!(!weave.exists(b));
        let c = weave.new_knot();
        assert!(weave.exists(c));
        assert!(a == c || b == c); // reuse happened
        assert!(weave.is_knot(c).unwrap());
        assert!(!weave.is_mark(c).unwrap());
        let d = weave.new_knot();
        assert!(weave.exists(d));
        assert_ne!(c, d);
        assert!(a == d || b == d); // reuse happened
    }
}