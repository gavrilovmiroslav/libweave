use std::collections::{HashMap, HashSet};
use array_tool::vec::Intersect;
use itertools::Itertools;
use ordered_multimap::ListOrderedMultimap;
use crate::embedding::FindAllEmbeddings;
use crate::weave::{Cover, Embedding, Weave, Weaveable};

#[derive(Debug, Default)]
pub struct PatternMatchingEmbedding {
    candidates: ListOrderedMultimap<usize, usize>,
    pattern_candidates: ListOrderedMultimap<usize, usize>,
    candidate_mapping: HashMap<usize, (usize, usize)>,
    rev_candidate_mapping: HashMap<(usize, usize), usize>,
    loops: HashMap<usize, usize>,
}

fn find_candidates_by_degrees(weave: &Weave, query: &Cover, data: &Cover) -> PatternMatchingEmbedding {
    let mut pattern_matching = PatternMatchingEmbedding::default();
    let mut in_degree_map = ListOrderedMultimap::new();
    let mut out_degree_map = ListOrderedMultimap::new();
    let mut loop_degree_map = ListOrderedMultimap::new();

    for target_node in &data.knots {
        let loop_degree = weave.get_loop_degree(*target_node).unwrap();
        let in_degree = weave.get_in_degree(*target_node).unwrap() - loop_degree;
        let out_degree = weave.get_out_degree(*target_node).unwrap() - loop_degree;
        
        for i in 0..=in_degree {
            in_degree_map.append(i, *target_node);
        }

        for i in 0..=out_degree {
            out_degree_map.append(i, *target_node);
        }

        for i in 0..=loop_degree {
            loop_degree_map.append(i, *target_node);
        }

        pattern_matching.loops.insert(*target_node, loop_degree);

        for pattern_node in &query.knots {
            let loops = weave.get_loops(*pattern_node);
            let loop_degree = loops.len();
            let in_degree = weave.get_in_degree(*pattern_node).unwrap() - loop_degree;
            let out_degree = weave.get_out_degree(*pattern_node).unwrap() - loop_degree;
            
            let in_candidates = in_degree_map.get_all(&in_degree).collect_vec();
            let out_candidates = out_degree_map.get_all(&out_degree).collect_vec();
            let loop_candidates = loop_degree_map.get_all(&loop_degree).collect_vec();

            in_candidates
                .intersect(out_candidates)
                .intersect(loop_candidates)
                .into_iter()
                .for_each(|target_node| {
                    pattern_matching.candidates.append(*pattern_node, *target_node);
                });
        }
    }

    pattern_matching
}

fn assign_candidate_and_test(
    weave: &Weave,
    pattern: &Cover,
    state: &PatternMatchingEmbedding,
    remaining_candidates: &[usize],
    bindings: &mut HashMap<usize, usize>,
    results: &mut Vec<HashMap<usize, usize>>,
) {
    if let Some((head, tail)) = remaining_candidates.split_first() {
        for binding in state.pattern_candidates.get_all(head) {
            bindings.insert(*head, *binding);
            assign_candidate_and_test(weave, pattern, state, tail, bindings, results);
            bindings.remove(head);
        }
    } else {
        let traversal = Cover::from(bindings.values().cloned().collect_vec());

        let candidates = find_candidates_by_degrees(weave, pattern, &traversal).candidates;
        let candidates_found = candidates.keys_len();
        
        if candidates_found == bindings.len() {
            results.push(HashMap::from_iter(
                bindings
                    .iter()
                    .map(|(k, v)| (*k, state.candidate_mapping.get(v).unwrap().1))
                    .collect_vec(),
            ));
        }
    }
}

impl FindAllEmbeddings for PatternMatchingEmbedding {
    fn find_all_embeddings(weave: &Weave, embed: usize, query: Cover, data: Cover) -> Vec<Embedding> {
        let mut embeddings = vec![];
        let mut state = find_candidates_by_degrees(weave, &query, &data);
        let mut transient = vec![];

        for start_node in &query.knots {
            let start_candidates = state.candidates.get_all(start_node).collect_vec();

            for &sc in &start_candidates {
                let candidate = weave.new_knot();

                state.candidate_mapping.insert(candidate, (*start_node, *sc));
                state.rev_candidate_mapping.insert((*start_node, *sc), candidate);
                state.pattern_candidates.append(*start_node, candidate);

                for _ in 0..*state.loops.get(sc).unwrap() {
                    weave.new_arrow(candidate, candidate);
                }

                transient.push(candidate);
            }
        }

        for start_node_in_pattern in &query.knots {
            let pid = *start_node_in_pattern;
            let start_candidates_in_target = state
                .candidates
                .get_all(start_node_in_pattern)
                .collect_vec();

            for end_node_in_pattern in weave.get_neighbors(pid) {
                let tid = end_node_in_pattern;
                let end_candidates_in_target = state
                    .candidates
                    .get_all(&end_node_in_pattern)
                    .collect_vec();

                for &start_candidate_in_target in &start_candidates_in_target {
                    for &end_candidate_in_target in &end_candidates_in_target {
                        if start_candidate_in_target == end_candidate_in_target {
                            continue;
                        }

                        if !weave.are_connected(*start_candidate_in_target, *end_candidate_in_target) {
                            continue;
                        }

                        let candidate1 = state.rev_candidate_mapping
                            .get(&(pid, *start_candidate_in_target))
                            .unwrap();

                        let candidate2 = state.rev_candidate_mapping
                            .get(&(tid, *end_candidate_in_target))
                            .unwrap();

                        let binding = weave.new_arrow(*candidate1, *candidate2).unwrap();
                        transient.push(binding);
                    }
                }
            }
        }

        let keys = state.pattern_candidates.keys().cloned().collect_vec();

        let mut results = Vec::new();
        assign_candidate_and_test(
            weave,
            &query,
            &state,
            &keys,
            &mut HashMap::new(),
            &mut results,
        );

        for result in results.clone() {
            let mut values = HashSet::new();

            for v in result.values() {
                values.insert(v);
            }

            if values.len() < result.len() {
                continue;
            }

            let mut embedding = Embedding { relation: embed, image: Default::default() };

            for (k, v) in &result {
                embedding.image.insert(*k, *v);
            }

            embeddings.push(embedding);
        }

        for t in transient {
            weave.delete(t);
        }

        embeddings
    }
}