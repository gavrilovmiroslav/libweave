use std::os::raw::c_void;
use itertools::Itertools;
use crate::weave::{MotifId, Weave, WeaveRef, Weaveable};

#[repr(C)]
pub struct IdVec {
    data: *mut c_void,
    len: usize,
}

#[repr(C)]
pub struct IdCover {
    hash: u64,
    knots: IdVec,
}

#[repr(C)]
pub struct IdEmbedding {
    size: usize,
    keys: IdVec,
    vals: IdVec,
}

impl From<Vec<i32>> for IdVec {
    fn from(vec: Vec<i32>) -> Self {
        let len = vec.len();
        let data = vec.as_ptr() as *mut c_void;
        std::mem::forget(vec);
        IdVec { data, len }
    }
}

impl From<Vec<usize>> for IdVec {
    fn from(vec: Vec<usize>) -> Self {
        let len = vec.len();
        let data = vec.as_ptr() as *mut c_void;
        std::mem::forget(vec);
        IdVec { data, len }
    }
}



#[no_mangle]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn weave_create<'s>() -> WeaveRef<'s> {
    Weave::create()
}

#[no_mangle]
pub extern "C" fn weave_source(weave: Weave, index: usize) -> usize {
    weave.get_source(index)
}

#[no_mangle]
pub extern "C" fn weave_source_nth(weave: Weave, index: usize, degree: usize) -> usize {
    weave.get_source_nth(index, degree)
}

#[no_mangle]
pub extern "C" fn weave_target(weave: Weave, index: usize) -> usize {
    weave.get_target(index)
}

#[no_mangle]
pub extern "C" fn weave_target_nth(weave: Weave, index: usize, degree: usize) -> usize {
    weave.get_target_nth(index, degree)
}

#[no_mangle]
pub extern "C" fn weave_new_knot(weave: Weave) -> usize {
    weave.new_knot()
}

#[no_mangle]
pub extern "C" fn weave_new_arrow(weave: Weave, source_index: usize, target_index: usize) -> usize {
    weave.new_arrow(source_index, target_index).unwrap_or(weave.bottom())
}

#[no_mangle]
pub extern "C" fn weave_new_tether(weave: Weave, source_index: usize) -> usize {
    weave.new_tether(source_index).unwrap_or(weave.bottom())
}

#[no_mangle]
pub extern "C" fn weave_new_mark(weave: Weave, target_index: usize) -> usize {
    weave.new_mark(target_index).unwrap_or(weave.bottom())
}

#[no_mangle]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn weave_identify(weave: Weave, index: usize) -> MotifId {
    weave.identify(index)
}

#[no_mangle]
pub extern "C" fn weave_is_knot(weave: Weave, index: usize) -> bool {
    weave.is_knot(index).unwrap_or(false)
}

#[no_mangle]
pub extern "C" fn weave_is_arrow(weave: Weave, index: usize) -> bool {
    weave.is_arrow(index).unwrap_or(false)
}

#[no_mangle]
pub extern "C" fn weave_is_tether(weave: Weave, index: usize) -> bool {
    weave.is_tether(index).unwrap_or(false)
}

#[no_mangle]
pub extern "C" fn weave_is_mark(weave: Weave, index: usize) -> bool {
    weave.is_mark(index).unwrap_or(false)
}

#[no_mangle]
pub extern "C" fn weave_are_ambi_connected(weave: Weave, source_index: usize, target_index: usize) -> bool {
    weave.are_ambi_connected(source_index, target_index)
}

#[no_mangle]
pub extern "C" fn weave_are_bi_connected(weave: Weave, source_index: usize, target_index: usize) -> bool {
    weave.are_bi_connected(source_index, target_index)
}

#[no_mangle]
pub extern "C" fn weave_are_connected(weave: Weave, source_index: usize, target_index: usize) -> bool {
    weave.are_connected(source_index, target_index)
}

#[no_mangle]
pub extern "C" fn weave_get_connections(weave: Weave, source_index: usize, target_index: usize) -> IdVec {
    IdVec::from(weave.get_connections(source_index, target_index))
}

#[no_mangle]
pub extern "C" fn weave_get_connections_from(weave: Weave, source_index: usize) -> IdVec {
    IdVec::from(weave.get_connections_from(source_index))
}

#[no_mangle]
pub extern "C" fn weave_get_connections_to(weave: Weave, target_index: usize) -> IdVec {
    IdVec::from(weave.get_connections_to(target_index))
}

#[no_mangle]
pub extern "C" fn weave_get_neighbors(weave: Weave, index: usize) -> IdVec {
    IdVec::from(weave.get_neighbors(index))
}

#[no_mangle]
pub extern "C" fn weave_get_co_neighbors(weave: Weave, index: usize) -> IdVec {
    IdVec::from(weave.get_co_neighbors(index))
}

#[no_mangle]
pub extern "C" fn weave_get_tethers(weave: Weave, index: usize) -> IdVec {
    IdVec::from(weave.get_tethers(index))
}

#[no_mangle]
pub extern "C" fn weave_get_marks(weave: Weave, index: usize) -> IdVec {
    IdVec::from(weave.get_marks(index))
}

#[no_mangle]
pub extern "C" fn weave_get_hoisted_arrows(weave: Weave, source_index: usize, target_index: usize) -> IdVec {
    IdVec::from(weave.get_hoisted_arrows(source_index, target_index))
}

#[no_mangle]
pub extern "C" fn weave_get_hoisted_arrows_from(weave: Weave, index: usize) -> IdVec {
    IdVec::from(weave.get_hoisted_arrows_from(index))
}

#[no_mangle]
pub extern "C" fn weave_get_hoisted_arrows_to(weave: Weave, index: usize) -> IdVec {
    IdVec::from(weave.get_hoisted_arrows_to(index))
}

#[no_mangle]
pub extern "C" fn weave_get_hoist_endpoints(weave: Weave, index: usize) -> IdVec {
    if let Some((source, target)) = weave.get_hoist_endpoints(index) {
        IdVec::from(vec![ source, target ])
    } else {
        IdVec::from(Vec::<usize>::new())
    }
}

#[no_mangle]
pub extern "C" fn weave_get_flow_graph_cover(weave: Weave, knot_index: usize) -> IdCover {
    let cover = weave.get_flow_graph_cover(knot_index);
    IdCover {
        hash: cover.hash,
        knots: IdVec::from(cover.knots),
    }
}

#[no_mangle]
pub extern "C" fn weave_get_graph_cover(weave: Weave, knot_index: usize) -> IdCover {
    let cover = weave.get_graph_cover(knot_index);
    IdCover {
        hash: cover.hash,
        knots: IdVec::from(cover.knots),
    }
}

#[no_mangle]
pub extern "C" fn weave_find_embeddings(weave: Weave, embed_relation: usize) -> IdEmbedding {
    let embeddings = weave.find_embeddings(embed_relation);
    let mut keys = Vec::<usize>::new();
    let mut vals = Vec::<usize>::new();
    let mut size = 0;

    if let Some(embeds) = embeddings {
        if !embeds.is_empty() {
            size = embeds.first().unwrap().image.len();
        }

        for embed in &embeds {
            for k in embed.image.keys().sorted() {
                keys.push(*k);
                vals.push(*embed.image.get(k).unwrap());
            }
        }
    }
    
    IdEmbedding {
        size,
        keys: IdVec::from(keys),
        vals: IdVec::from(vals),
    }
}