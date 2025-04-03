use crate::weave::{MotifIdx};

pub trait WeaveExtension<T> {
    fn ext_to_motif(t: T) -> MotifIdx;
    fn motif_to_ext(motif: MotifIdx) -> T;
}

