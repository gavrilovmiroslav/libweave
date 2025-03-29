use std::cell::RefCell;
use std::collections::HashMap;
use id_arena::{Arena, Id};
use multimap::MultiMap;

pub type MotifId = Id<Motif>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Motif {
    Knot,
    Arrow { source: MotifId, target: MotifId },
    Tether { source: MotifId },
    Mark { target: MotifId },
}

#[repr(C)]
#[derive(Debug)]
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

#[allow(dead_code)]
pub trait Weaveable<W> {
    fn create() -> W;
    fn get_source(&self, index: usize) -> usize;
    fn get_source_nth(&self, index: usize, degree: usize) -> usize;
    fn get_target(&self, index: usize) -> usize;
    fn get_target_nth(&self, index: usize, degree: usize) -> usize;
    fn new_knot(&self) -> usize;
    fn new_arrow(&self, source: usize, target: usize) -> Option<usize>;
    fn new_tether(&self, source: usize) -> Option<usize>;
    fn new_mark(&self, target: usize) -> Option<usize>;
    fn identify(&self, index: usize) -> MotifId;
    fn opt_identify(&self, index: usize) -> Option<MotifId>;
    fn is_knot(&self, index: usize) -> Option<bool>;
    fn is_arrow(&self, index: usize) -> Option<bool>;
    fn is_tether(&self, index: usize) -> Option<bool>;
    fn is_mark(&self, index: usize) -> Option<bool>;
    fn are_ambi_connected(&self, source: usize, target: usize) -> bool;
    fn are_bi_connected(&self, source: usize, target: usize) -> bool;
    fn are_connected(&self, source: usize, target: usize) -> bool;
    fn get_connections(&self, source: usize, target: usize) -> Vec<usize>;
    fn get_connections_from(&self, source: usize) -> Vec<usize>;
    fn get_connections_to(&self, target: usize) -> Vec<usize>;
    fn get_neighbors(&self, index: usize) -> Vec<usize>;
    fn get_co_neighbors(&self, index: usize) -> Vec<usize>;
    fn get_tethers(&self, index: usize) -> Vec<usize>;
    fn get_co_tethers(&self, index: usize) -> Vec<usize>;
    fn get_marks(&self, index: usize) -> Vec<usize>;
    fn get_co_marks(&self, index: usize) -> Vec<usize>;
}

#[repr(C)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
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
        if internal.motif_co_neighbors.contains_key(&source) {
            for neighbor in internal.motif_co_neighbors.get_vec(&source).unwrap() {
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

    fn get_co_tethers(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_tethers
            .iter().flat_map(|(id, tether)|
                if *tether == index { vec![*id] } else { vec![] })
            .collect::<Vec<usize>>()
    }

    fn get_marks(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_marks.get_vec(&index).unwrap_or(&vec![]).to_vec()
    }

    fn get_co_marks(&self, index: usize) -> Vec<usize> {
        let internal = self.0.borrow();
        internal.motif_marks
            .iter().flat_map(|(id, mark)|
            if *mark == index { vec![*id] } else { vec![] })
            .collect::<Vec<usize>>()
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
        assert_eq!(weave.get_co_marks(m), vec![ a ]);
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
        assert_eq!(weave.get_co_tethers(t), vec![ a ]);
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
    fn test_hierarchy_design() {
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
    }
}