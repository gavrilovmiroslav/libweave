use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hasher};
use itertools::Itertools;
use crate::weave::{Cover, Embedding, Weave, Weaveable};

struct SearchEmbeddingContext<'w, 's> {
    weave: Weave<'w, 's>,
    embed: usize,
    query: Cover,
    data: Cover,
}

#[derive(Debug, Clone)]
struct CandidateSet {
    bindings: HashMap<usize, Vec<usize>>,
}

type DeadEndPattern = HashMap::<(usize, usize), HashSet<(usize, usize)>>;
impl From<&'_ SearchEmbeddingContext<'_, '_>> for CandidateSet {
    fn from(value: &SearchEmbeddingContext) -> Self {
        let mut hm: HashMap<usize, Vec<usize>> = HashMap::new();
        for k in value.query.knots.iter() {
            hm.insert(*k, value.data.knots.clone());
        }

        CandidateSet {
            bindings: hm,
        }
    }
}

pub fn find_all_embeddings(weave: &Weave, embed: usize, query: Cover, data: Cover) -> Vec<Embedding> {
    let context = SearchEmbeddingContext { weave, embed, query, data };
    let mut embeddings = Vec::<Embedding>::new();
    let mut gamma = HashSet::<usize>::new();
    let mut delta = DeadEndPattern::new();
    let mut partial_map = HashMap::new();
    partial_map.insert(*context.query.knots.first().unwrap(), *context.data.knots.first().unwrap());
    search(0, &context, &mut embeddings, &mut gamma, &mut delta,
           partial_map, CandidateSet::from(&context));

    embeddings
}

fn search(id: u64, context: &SearchEmbeddingContext, embeddings: &mut Vec<Embedding>,
          gamma: &mut HashSet<usize>, delta: &mut DeadEndPattern,
          partial_map: HashMap<usize, usize>, candidate_set: CandidateSet) -> HashSet<usize> {
    println!("SEARCH with ID = {}", id);

    // 2
    let k = partial_map.len();
    // 3
    if k == context.data.knots.len() {
        // 4
        embeddings.push(create_embedding(context, &partial_map));
        HashSet::default()
    } else {
        // 6
        let new_candidate_set = refine_candidate_set_with_edge_constraints(context.weave, &candidate_set, &partial_map);
        println!("Candidate set for {} = {:?}", id, new_candidate_set);
        // for line 19
        let past_report = embeddings.len();

        // 7
        for (ui, v) in &new_candidate_set.bindings {
            if v.is_empty() {
                // 8
                for n in context.weave.get_ambi_neighbors(*ui) {
                    if partial_map.contains_key(&n) {
                        gamma.insert(n);
                    }
                }
            } else {
                // 10
                let mut g = HashSet::<usize>::new();
                // 11
                let candidates = new_candidate_set.bindings.get(&(k + 1)).unwrap_or(&vec![]).to_vec();
                for v in &candidates {
                    // 12
                    if partial_map.contains_key(v) {
                        // 13
                        for (m, mv) in &partial_map {
                            if *mv == *v { g.insert(*m); }
                        }
                    } else {
                        // 14
                        if let Some(ukp1) = context.query.knots.get(k + 1) {
                            let mut subset = true;
                            if let Some(de) = delta.get(&(*ukp1, *v)) {
                                for (du, dv) in de {
                                    if !partial_map.contains_key(du) || *partial_map.get(du).unwrap() != *dv {
                                        subset = false;
                                        break;
                                    }
                                }

                                if subset {
                                    // 15
                                    for (du, _dv) in de {
                                        g.insert(*du);
                                    }
                                }
                            }

                            if !subset {
                                // 17
                                let mut pm = HashMap::new();
                                for (a, b) in &partial_map {
                                    pm.insert(*a, *b);
                                }
                                pm.insert(*ukp1, *v);
                                // 17.5
                                let cs = candidate_set.clone();
                                g.extend(search(get_id(&pm), context, embeddings, gamma, delta, pm, cs));
                            }
                        }
                    }
                }
                // 18
                gamma.extend(g);
            }
        }

        let new_report = embeddings.len();
        // 19
        if new_report != past_report && !partial_map.is_empty() {
            // 20
            for (ui, v) in &partial_map {
                let mut diff = vec![];
                if gamma.contains(ui) {
                    diff.push((*ui, *v));
                }
                let uk = *context.query.knots.get(k).unwrap();
                let muk = *partial_map.get(&uk).unwrap();
                delta.insert((uk, muk), HashSet::from_iter(diff.into_iter()));
            }

            // 21
            return gamma.clone();
        }

        // 22
        HashSet::default()
    }
}

fn get_id(partial_map: &HashMap<usize, usize>) -> u64 {
    let mut hasher = DefaultHasher::default();
    for k in partial_map.keys().sorted() {
        hasher.write_usize(*k);
    }

    hasher.finish()
}

fn refine_candidate_set_with_edge_constraints(weave: Weave, candidate_set: &CandidateSet, partial_map: &HashMap<usize, usize>) -> CandidateSet {
    let mut new_bindings = HashMap::new();

    for (ui, cui) in candidate_set.bindings.iter() {
        let mut stay = HashSet::new();
        for neighbor in weave.get_ambi_neighbors(*ui) {
            // in here: N(ui) /intersect dom(M), neighbor == ui'
            if let Some(n) = partial_map.get(&neighbor) {
                for neighbor in weave.get_ambi_neighbors(*n) {
                    if cui.contains(&neighbor) {
                        stay.insert(neighbor);
                    }
                }
            }
        }

        new_bindings.insert(*ui, stay.iter().copied().collect::<Vec<usize>>());
    }

    CandidateSet {
        bindings: new_bindings
    }
}

fn create_embedding(context: &SearchEmbeddingContext, partial_map: &HashMap<usize, usize>) -> Embedding {
    Embedding {
        image: partial_map.clone(),
        relation: context.embed,
    }
}