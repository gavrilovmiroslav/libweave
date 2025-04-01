use crate::weave::{Cover, Embedding, Weave};

pub struct SearchEmbeddingContext<'w, 's> {
    pub(crate) weave: Weave<'w, 's>,
    pub(crate) embed: usize,
    pub(crate) query: Cover,
    pub(crate) data: Cover,
}

pub trait FindAllEmbeddings {
    fn find_all_embeddings(weave: &Weave, embed: usize, query: Cover, data: Cover) -> Vec<Embedding>;
}