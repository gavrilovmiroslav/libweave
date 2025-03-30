use std::cell::RefCell;
use std::collections::HashMap;
use id_arena::{Arena, Id};
use multimap::MultiMap;

/// Type alias for motif ID in libweave's internal arenas.
pub type MotifId = Id<Motif>;

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
/// of an non-existent motif ID).
#[repr(C)]
#[derive(Clone, Debug)]
pub struct WeaveInternal<'s> {
    pub(crate) motif_bottom: usize,
    motif_space: Arena<Motif>,
    motif_index: HashMap<usize, MotifId>,
    motif_conns: MultiMap<(usize, usize), usize>,
    motif_neighbors: MultiMap<usize, usize>,
    motif_co_neighbors: MultiMap<usize, usize>,
    motif_tethers: MultiMap<usize, usize>,
    motif_marks: MultiMap<usize, usize>,
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
            motif_conns: Default::default(),
            motif_neighbors: Default::default(),
            motif_co_neighbors: Default::default(),
            motif_tethers: Default::default(),
            motif_marks: Default::default(),
            string_space: Default::default(),
        }
    }
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
    ///     [a] ---->(c) >---e---> (d)----> [b]
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
    fn get_source(&self, index: usize) -> usize;

    /// Gets the n-th degree **source** of a `motif`. This is equivalent to calling `get_source`
    /// `n` times in a sequence, passing the result of the previous query as the argument.
    ///
    /// # Example
    ///
    ///     get_source_nth(e, 2) = get_source(get_source(e))
    ///
    fn get_source_nth(&self, index: usize, degree: usize) -> usize;

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
    fn get_target(&self, index: usize) -> usize;

    /// Gets the n-th degree **target** of a `motif`. This is equivalent to calling `get_target`
    /// `n` times in a sequence, passing the result of the previous query as the argument.
    ///
    /// # Example
    ///
    ///     get_target_nth(e, 2) = get_target(get_target(e))
    ///
    fn get_target_nth(&self, index: usize, degree: usize) -> usize;

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
    /// assert!(weave.is_knot(a));
    /// ```
    fn new_knot(&self) -> usize;

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
    /// assert!(weave.is_arrow(c));
    /// ```
    fn new_arrow(&self, source: usize, target: usize) -> Option<usize>;

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
    /// assert!(weave.is_tether(b));
    /// ```
    fn new_tether(&self, source: usize) -> Option<usize>;

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
    /// assert!(weave.is_mark(b));
    /// ```
    fn new_mark(&self, target: usize) -> Option<usize>;

    /// Identifies the underlying `MotifId` for the given `index`. Mostly internal use.
    fn identify(&self, index: usize) -> MotifId;

    /// Identifies the underlying `MotifId` for the given `index` but instead of returning
    /// the `bottom` motif, it returns `None` when `index` doesn't exist. Mostly internal use.
    fn opt_identify(&self, index: usize) -> Option<MotifId>;

    /// Returns an option with `true` if the motif under `index` is a knot. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_knot(&self, index: usize) -> Option<bool>;

    /// Returns an option with `true` if the motif under `index` is an arrow. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_arrow(&self, index: usize) -> Option<bool>;

    /// Returns an option with `true` if the motif under `index` is a tether. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_tether(&self, index: usize) -> Option<bool>;

    /// Returns an option with `true` if the motif under `index` is a mark. Returns `None` if the
    /// motif under `index` doesn't exist.
    fn is_mark(&self, index: usize) -> Option<bool>;

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
    fn are_ambi_connected(&self, source: usize, target: usize) -> bool;

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
    fn are_bi_connected(&self, source: usize, target: usize) -> bool;

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
    /// assert!(!weave.are_connected(a, b));
    /// assert!(!weave.are_connected(b, a));
    /// ```
    fn are_connected(&self, source: usize, target: usize) -> bool;

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
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// let d = weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_connections(a, b), vec![ c ]);
    /// let e = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_connections(a, b), vec![ c, e ]);
    /// ```
    fn get_connections(&self, source: usize, target: usize) -> Vec<usize>;

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
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// let d = weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// let e = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_connections_from(a), vec![ c, e ]);
    /// ```
    fn get_connections_from(&self, source: usize) -> Vec<usize>;

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
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// let d = weave.new_arrow(b, a).unwrap_or(weave.bottom());
    /// let e = weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_connections_to(b), vec![ c, e ]);
    /// ```
    fn get_connections_to(&self, target: usize) -> Vec<usize>;

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
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_knot();
    /// let c = weave.new_knot();
    /// let d = weave.new_knot();
    /// weave.new_arrow(a, b).unwrap_or(weave.bottom());
    /// weave.new_arrow(a, c).unwrap_or(weave.bottom());
    /// weave.new_arrow(d, a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_neighbors(a), vec![ b, c ]);
    /// ```
    fn get_neighbors(&self, index: usize) -> Vec<usize>;

    /// Gets the immediate co-neighbors of this motif. If `index` is malformed, the result is an
    /// empty container. For the purposes of this function, a co-neighbor is the dual of a neighbor:
    /// if one motif is a neighbor to another, then the other is a co-neighbor to the first.
    ///
    /// # Example
    ///
    /// ```plaintext
    ///            pre                        post
    ///  -----------------------------------------------------
    ///       [a] >----> [b]         coneighbors(b) = { a, c }
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
    /// assert_eq!(weave.get_neighbors(b), vec![ a, c ]);
    /// ```
    fn get_co_neighbors(&self, index: usize) -> Vec<usize>;

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
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_tether(a).unwrap_or(weave.bottom());
    /// let c = weave.new_tether(a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_tethers(a), vec![ b, c ]);
    /// ```
    fn get_tethers(&self, index: usize) -> Vec<usize>;

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
    /// use libweave::weave::{Weave, Weaveable};
    /// let weave = &Weave::create();
    /// let a = weave.new_knot();
    /// let b = weave.new_mark(a).unwrap_or(weave.bottom());
    /// let c = weave.new_mark(a).unwrap_or(weave.bottom());
    /// assert_eq!(weave.get_marks(a), vec![ b, c ]);
    /// ```
    fn get_marks(&self, index: usize) -> Vec<usize>;

    /// Gets all the hoisted arrows going from the `index` motif. A *hoisted arrow* is a specific
    /// construction in the form of `[a] --->(b) >---c---> (d)---> [e]`, comprising a tether, arrow,
    /// and mark. If this construction is found, `c` is referred to as being hoisted between `[a]`
    /// and `[e]` by the meta-motifs `(b)` and `(d)`. Hoisted arrows are useful to represent arrows
    /// in a meta structure, that isn't a part of some graph (for example, hierarchies).
    fn get_hoisted_arrows(&self, index: usize) -> Vec<usize>;

    /// Gets all the co-hoisted arrows going from the `index` motif, which is the same as getting
    /// all the *hoisted* arrows going _into_ this motif. A *hoisted arrow* is a specific construction
    /// in the form of `[a] --->(b) >---c---> (d)---> [e]`, comprising a tether, arrow, and mark.
    /// If this construction is found, `c` is referred to as being hoisted between `[a]` and `[e]`
    /// by the meta-motifs `(b)` and `(d)`. This is simply searching for hoisted arrows from the
    /// target element (`[e]` in the diagram).
    fn get_co_hoisted_arrows(&self, index: usize) -> Vec<usize>;
}

#[repr(C)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
#[derive(Clone, Debug)]
pub struct WeaveRef<'s>(pub(crate) Box<RefCell<WeaveInternal<'s>>>, pub(crate) usize);

pub type Weave<'w, 's> = &'w WeaveRef<'s>;

impl<'s> WeaveRef<'s> {
    pub fn bottom(&self) -> usize { self.1 }
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
        let knot = internal.motif_space.alloc(Motif::Knot);
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
            let arrow = internal.motif_space.alloc(Motif::Arrow { source, target });
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
            let tether = internal.motif_space.alloc(Motif::Tether { source });
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
            let mark = internal.motif_space.alloc(Motif::Mark { target });
            let index = mark.index();
            internal.motif_index.insert(index, mark);
            internal.motif_marks.insert(target_index, index);
            Some(index)
        } else {
            None
        }
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

    fn get_connections(&self, source: usize, target: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_conns.get_vec(&(source, target)).cloned().unwrap().to_vec()
    }

    fn get_connections_from(&self, source: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        let mut results = vec![];
        if internal.motif_neighbors.contains_key(&source) {
            for neighbor in internal.motif_neighbors.get_vec(&source).unwrap() {
                results.extend_from_slice(&internal.motif_conns.get_vec(&(source, *neighbor)).cloned().unwrap());
            }
        }

        results
    }

    fn get_connections_to(&self, target: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        let mut results = vec![];
        if internal.motif_co_neighbors.contains_key(&target) {
            for neighbor in internal.motif_co_neighbors.get_vec(&target).unwrap() {
                results.extend_from_slice(&internal.motif_conns.get_vec(&(*neighbor, target)).cloned().unwrap());
            }
        }

        results
    }

    fn get_neighbors(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_neighbors.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_co_neighbors(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_co_neighbors.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_tethers(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_tethers.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_marks(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_marks.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_hoisted_arrows(&self, index: usize) -> Vec<usize> {
        self.get_tethers(index).iter()
            .flat_map(|t| self.get_connections_from(*t))
            .filter(|a| self.is_mark(self.get_target(*a)).unwrap_or(false))
            .collect()
    }

    fn get_co_hoisted_arrows(&self, index: usize) -> Vec<usize> {
        self.get_marks(index).iter()
            .flat_map(|t| self.get_connections_to(*t))
            .filter(|a| self.is_tether(self.get_source(*a)).unwrap_or(false))
            .collect()
    }
}

#[cfg(test)]
mod tests {
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

    /*
        Ambi-connected: connected with at least one arrow in any direction
        Bi-connected: connected with at least one arrow in both directions

        ambi(x, y) == ambi(x, y)  and  bi(x, y) == bi(y, x)

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
        They are knot-like in the fact that they are ENDPOINTs.
        They are arrow-like in the fact that they "tie" to other motifs.
        Marks are good form for making properties of objects.
     */
    #[test]
    fn test_mark() {
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
        They are knot-like in the fact that they are ENDPOINTs.
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
        Hierarchies are arrows between a tether and a mark,
        grounding objects but not bulking their arrow-sets
        with meta-arrows that aren't an *actual* part of a
        graph.

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

        fn get_child_and_parent(weave: Weave, parenthood: usize) -> Option<(usize, usize)> {
            if weave.is_arrow(parenthood).unwrap_or(false) {
                let (source, target) =
                    (weave.get_source_nth(parenthood, 2),
                     weave.get_target_nth(parenthood, 2));

                return Some((source, target));
            }

            None
        }

        let weave = &Weave::create();
        let a = weave.new_knot();
        let b = weave.new_knot();
        let parenthood = make_parent_hierarchy(weave, a, b).unwrap();
        let cp = get_child_and_parent(weave, parenthood);

        assert!(cp.is_some());
        let (child, parent) = cp.unwrap();
        assert_eq!(child, a);
        assert_eq!(parent, b);

        assert_eq!(weave.get_hoisted_arrows(a), vec![ parenthood ]);
        assert_eq!(weave.get_co_hoisted_arrows(b), vec![ parenthood ]);
    }
}