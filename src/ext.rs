use std::fmt::Debug;
use crate::weave::{MotifIdx};

/// Weave extensions let us tie a `Weaveable` to an external API that might have its own identifier 
/// system. Weave itself doesn't couple itself with any of these extensions, and this mechanism can
/// be implemented _within_ the external API as well.
pub trait WeaveExtension<T: Debug + Clone + Copy> {
    /// Transforms the external identifier to a motif index `MotifIdx`
    fn ext_to_motif(t: T) -> MotifIdx;
    
    /// Transforms a `MotifIdx` into an external identifier.
    fn motif_to_ext(motif: MotifIdx) -> T;
}

