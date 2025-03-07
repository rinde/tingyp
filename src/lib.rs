//! A tiny genetic programming library.
//!
//! # Example
//!
//! ```
//! use nonzero_lit::usize;
//! use tinygp::NodeType::*;
//! use tinygp::*;
//! let generator = WeightedNodeGenerator::new(
//!     [(Add, 1), (Sub, 1), (Const(1.0), 1), (Var(Vars::X), 2)],
//!     [(Const(1.0), 1), (Var(Vars::X), 1)],
//! );
//!
//! let population_size = usize!(200);
//! let max_tree_depth = usize!(6);
//! let random_seed = 0;
//! let mut evolver = Evolver::new(population_size, max_tree_depth, generator, random_seed);
//!
//! let generations = 10;
//! let (best, best_fitness) = evolver.evolve(&Sub3, generations);
//!
//! assert!(
//!     best_fitness <= f64::EPSILON,
//!     "A perfect program was not found, best fitness {}",
//!     best_fitness
//! );
//!
//! #[derive(Debug, Clone, Copy, PartialEq, Eq)]
//! enum Vars {
//!     X,
//! }
//!
//! #[derive(Default, Clone)]
//! struct Context {
//!     x: f64,
//! }
//!
//! impl Variable for Vars {
//!     type Context = Context;
//!     type Value = f64;
//!
//!     fn name(&self) -> &'static str {
//!         match self {
//!             Self::X => "x",
//!         }
//!     }
//!
//!     fn value(&self, context: &Self::Context) -> f64 {
//!         match self {
//!             Self::X => context.x,
//!         }
//!     }
//! }
//!
//! /// An evaluator that calculates how closely a Tree matches the function `x - 3`.
//! struct Sub3;
//!
//! impl Evaluator<Vars> for Sub3 {
//!     fn evaluate(&self, _gen: usize, _last_gen: bool, individual: &Tree<Vars>) -> f64 {
//!         let mut error_sum = 0.0;
//!         for x in 0..100 {
//!             let x = f64::from(x);
//!             let expected_y = x - 3.0;
//!             let context = Context { x };
//!             let actual_y = individual.eval(&context);
//!             error_sum += (actual_y - expected_y).abs();
//!         }
//!         error_sum
//!     }
//! }
//! ```
use std::array;
use std::fmt::Display;
use std::hash::Hash;
use std::iter::Sum;
use std::num::NonZeroUsize;
use std::num::Saturating;
use std::ops::Neg;

use educe::Educe;
use nonempty_collections::IntoIteratorExt;
use nonempty_collections::NEVec;
use nonempty_collections::NonEmptyIterator;
use num::Float;
use num::Num;
use num::One;
use num::ToPrimitive;
use num::Zero;
use num::traits::ConstOne;
use num::traits::ConstZero;
use rand::Rng;
use rand::SeedableRng;
use rand::seq::index;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use strum::EnumDiscriminants;
use strum::EnumMessage;
use strum::EnumString;
use strum::VariantArray;

pub type Random = Xoshiro256PlusPlus;

#[derive(Debug, Clone)]
pub struct Evolver<V: Variable, G> {
    population: NEVec<Tree<V>>,
    max_tree_depth: NonZeroUsize,
    generator: G,
    rng: Random,
}

impl<V, G> Evolver<V, G>
where
    V: Variable,
    G: RandomNodeGenerator<V>,
{
    #[expect(clippy::missing_panics_doc, reason = "false positive")]
    pub fn new(
        pop_size: NonZeroUsize,
        max_tree_depth: NonZeroUsize,
        generator: G,
        seed: u64,
    ) -> Self {
        let mut rng = Random::seed_from_u64(seed);
        Self {
            population: std::iter::repeat_with(|| {
                Tree::new_random(max_tree_depth.get(), &generator, &mut rng)
            })
            .try_into_nonempty_iter()
            .unwrap() // TODO this can be avoided with an update to nonempty-collections
            .take(pop_size)
            .collect(),
            max_tree_depth,
            generator,
            rng,
        }
    }

    #[expect(clippy::missing_panics_doc, reason = "false positive")]
    pub fn evolve(
        &mut self,
        evaluator: &impl Evaluator<V>,
        generations: usize,
    ) -> (Tree<V>, V::Value) {
        let mut generation_best = (Tree::default(), V::Value::MAX);

        let last_gen = generations - 1;
        for generation in 0..generations {
            println!("{generation}");

            let pop2 = self.population.iter().enumerate().collect::<Vec<_>>();

            let fitness = pop2
                .par_iter()
                .enumerate()
                .map(|(i, (i2, tree))| {
                    assert_eq!(i, *i2);
                    (
                        i,
                        evaluator.evaluate(generation, generation == last_gen, tree),
                    )
                })
                .collect::<Vec<_>>(); // TODO with rayon support for nonempty-collections we could collect into a NEVec here
            debug_assert!(fitness.is_sorted_by_key(|x| x.0));

            let sum = fitness
                .iter()
                .map(|(_, f)| *f)
                .sum::<V::Value>()
                .to_f64()
                .unwrap();
            println!(
                "fitness min {} avg {}",
                fitness
                    .iter()
                    .map(|(_, f)| f)
                    .min_by(|l, r| l
                        .partial_cmp(r)
                        .unwrap_or_else(|| panic!("{l:?} {r:?} comparison failed")))
                    .unwrap(),
                sum / fitness.len() as f64
            );

            // println!("{fitness:?}");

            let best_index = fitness
                .iter()
                .min_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap())
                .map(|(i, _)| i)
                .copied()
                .unwrap();

            // elitism 1
            let mut next_generation = NEVec::with_capacity(
                self.population.capacity(),
                self.population[best_index].clone(),
            );

            generation_best = (self.population[best_index].clone(), fitness[best_index].1);

            // if fitness[best_index] < all_time_best.1 {
            //     println!(" >>>> found better {}", fitness[best_index]);
            //     all_time_best = (self.population[best_index].clone(),
            // fitness[best_index]);

            //     if fitness[best_index].is_zero() {
            //         println!("Found optimum");
            //         return all_time_best;
            //     }
            // }

            // crossover 90%
            let crossover_target = (0.9 * (self.population.len().get() as f64)) as usize;
            while next_generation.len().get() < crossover_target {
                let mut p1 =
                    self.population[Self::select_by_tournament(&fitness, &mut self.rng)].clone();
                let mut p2 =
                    self.population[Self::select_by_tournament(&fitness, &mut self.rng)].clone();
                p1.crossover(&mut p2, &mut self.rng);

                next_generation.extend(
                    [p1, p2]
                        .into_iter()
                        .filter(|p| p.depth() < self.max_tree_depth.get()),
                );
            }

            // mutation 10% (rest)
            while next_generation.len() < self.population.len() {
                let mut mutated =
                    self.population[Self::select_by_tournament(&fitness, &mut self.rng)].clone();
                mutated.mutate(self.max_tree_depth.get(), &self.generator, &mut self.rng);
                next_generation.push(mutated);
            }

            self.population = next_generation;
        }
        generation_best
    }

    fn select_by_tournament(fitness: &[(usize, V::Value)], rng: &mut impl Rng) -> usize {
        index::sample(rng, fitness.len(), 7)
            .iter()
            .min_by(|&l, &r| fitness[l].partial_cmp(&fitness[r]).unwrap())
            .unwrap()
    }
}

pub trait Evaluator<V: Variable>: Send + Sync {
    fn evaluate(&self, generation: usize, last: bool, individual: &Tree<V>) -> V::Value;
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum NodeType<V: Variable> {
    If4,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Min,
    Max,
    Const(V::Value),
    Var(V),
}

pub trait RandomNodeGenerator<V: Variable> {
    fn generate(&self, rng: &mut impl Rng) -> NodeType<V>;
    fn generate_no_children(&self, rng: &mut impl Rng) -> NodeType<V>;
}

#[derive(Debug, Clone)]
pub struct WeightedNodeGenerator<const L: usize, const N: usize, V: Variable> {
    all: [(NodeType<V>, u8); L],
    leafs: [(NodeType<V>, u8); N],
}

impl<const L: usize, const N: usize, V: Variable> WeightedNodeGenerator<L, N, V> {
    /// Creates a new [`WeightedNodeGenerator`] with the specified weights.
    ///
    /// # Panics
    /// When `leafs` contains a [`NodeType`] that can have children.
    pub fn new(all: [(NodeType<V>, u8); L], leafs: [(NodeType<V>, u8); N]) -> Self {
        assert!(L > 0 && N > 0);
        for (n, _) in leafs {
            assert!(matches!(n, NodeType::Const(_) | NodeType::Var(_)));
        }
        Self { all, leafs }
    }
}

impl<const L: usize, const N: usize, V: Variable> RandomNodeGenerator<V>
    for WeightedNodeGenerator<L, N, V>
{
    fn generate(&self, rng: &mut impl Rng) -> NodeType<V> {
        self.all[rng.random_range(0..L)].0
    }

    fn generate_no_children(&self, rng: &mut impl Rng) -> NodeType<V> {
        self.leafs[rng.random_range(0..N)].0
    }
}

pub trait Variable: Clone + Copy + Eq + PartialEq + std::fmt::Debug + Send + Sync {
    type Value: Value + Send + Sync;

    type Context: Send + Sync;

    fn name(&self) -> &'static str;

    fn value(&self, context: &Self::Context) -> Self::Value;
}

pub trait Value:
    Num
    + Neg<Output = Self>
    + PartialOrd
    + Clone
    + Copy
    + ConstZero
    + ConstOne
    + Display
    + Sum
    + ToPrimitive
    + std::fmt::Debug
{
    const MAX: Self;
    const MIN: Self;

    #[must_use]
    fn min(self, other: Self) -> Self;
    #[must_use]
    fn max(self, other: Self) -> Self;
    #[must_use]
    fn saturating_add(self, other: Self) -> Self;
    #[must_use]
    fn saturating_sub(self, other: Self) -> Self;
    #[must_use]
    fn saturating_mul(self, other: Self) -> Self;
    #[must_use]
    fn saturating_div(self, other: Self) -> Self;
    #[must_use]
    fn saturating_neg(self) -> Self;
    #[must_use]
    fn double(self) -> Self;
    #[must_use]
    fn is_zero(self) -> bool;
    #[must_use]
    fn handle_infinity(value: Self) -> Self;
}

#[derive(Debug, Clone, Educe, Serialize, Deserialize)]
#[educe(Default)]
#[serde(bound = "V: Serialize + DeserializeOwned, V::Value: Serialize + DeserializeOwned")]
pub struct Tree<V: Variable> {
    #[educe(Default = Node::Const(V::Value::ZERO))]
    root: Node<V>,
}

impl<V: Variable> Tree<V> {
    fn new_random(
        max_depth: usize,
        generator: &impl RandomNodeGenerator<V>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            root: Node::<V>::grow(generator, rng, 0, max_depth),
        }
    }

    fn mutate(
        &mut self,
        max_depth: usize,
        generator: &impl RandomNodeGenerator<V>,
        rng: &mut impl Rng,
    ) {
        let (node, depth) = Self::pick_random(&mut self.root, 0, rng);
        *node = Node::<V>::grow(generator, rng, 0, max_depth - depth);
    }

    fn crossover(&mut self, other: &mut Self, rng: &mut impl Rng) {
        let (one, _) = Tree::pick_random(&mut self.root, 0, rng);
        let (two, _) = Tree::pick_random(&mut other.root, 0, rng);
        std::mem::swap(one, two);
    }

    fn pick_random<'a>(
        node: &'a mut Node<V>,
        depth: usize,
        rng: &mut impl Rng,
    ) -> (&'a mut Node<V>, usize) {
        if !node.has_children() {
            return (node, depth);
        }
        let children = node.children_mut().unwrap();
        let chosen = &mut children[rng.random_range(0..children.len())];
        if rng.random_bool(0.5) {
            // pick this one
            (chosen, depth)
        } else {
            // go deeper
            Self::pick_random(chosen, depth + 1, rng)
        }
    }

    pub fn eval(&self, ctx: &V::Context) -> V::Value {
        self.root.eval(ctx)
    }

    fn depth(&self) -> usize {
        let mut depth = 0;
        let mut children = self.root.children().collect::<Vec<_>>();
        loop {
            if children.is_empty() {
                return depth;
            }
            children = children.iter().flat_map(|c| c.children()).collect();
            depth += 1;
        }
    }

    pub fn to_rust(&self) -> String {
        format!("let result = {};", self.root.to_rust())
    }

    pub fn simplify(&mut self) {
        self.root.simplify();
    }

    pub fn to_dot(&self) -> String {
        let mut id = 0;
        format!("digraph {{\n{}}}", self.root.to_dot(&mut id))
    }
}

#[derive(Debug, EnumDiscriminants, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[strum_discriminants(derive(EnumString, EnumMessage, VariantArray, Hash))]
#[serde(bound = "V: Serialize + DeserializeOwned, V::Value: Serialize + DeserializeOwned")]
enum Node<V: Variable> {
    If4([Box<Node<V>>; 4]),
    Add([Box<Node<V>>; 2]),
    Sub([Box<Node<V>>; 2]),
    Mul([Box<Node<V>>; 2]),
    Div([Box<Node<V>>; 2]),

    Neg([Box<Node<V>>; 1]),
    Min([Box<Node<V>>; 2]),
    Max([Box<Node<V>>; 2]),
    Const(V::Value),
    Var(V),
}

impl<V: Variable> Node<V> {
    const fn has_children(&self) -> bool {
        use Node::*;
        match self {
            If4(_) | Add(_) | Sub(_) | Mul(_) | Div(_) | Neg(_) | Min(_) | Max(_) => true,
            Const(_) | Var(_) => false,
        }
    }

    fn children(&self) -> impl Iterator<Item = &Box<Node<V>>> {
        use Node::*;
        match self {
            If4(children) => children.iter(),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => children.iter(),
            Neg(children) => children.iter(),
            Const(_) | Var(_) => [].iter(),
        }
    }

    fn children_mut(&mut self) -> Option<&mut [Box<Node<V>>]> {
        use Node::*;
        match self {
            If4(children) => Some(children),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => Some(children),
            Neg(children) => Some(children),
            Const(_) | Var(_) => None,
        }
    }

    fn eval(&self, ctx: &V::Context) -> V::Value {
        use Node::*;
        match self {
            If4(children) => {
                if children[0].eval(ctx) < children[1].eval(ctx) {
                    children[2].eval(ctx)
                } else {
                    children[3].eval(ctx)
                }
            }
            Add(children) => children[0].eval(ctx).saturating_add(children[1].eval(ctx)),
            Sub(children) => children[0].eval(ctx).saturating_sub(children[1].eval(ctx)),
            Mul(children) => children[0].eval(ctx).saturating_mul(children[1].eval(ctx)),
            Div(children) => {
                let v1 = children[1].eval(ctx);
                if v1.is_zero() {
                    V::Value::MAX
                } else {
                    children[0].eval(ctx).saturating_div(v1)
                }
            }
            Neg(children) => children[0].eval(ctx).saturating_neg(),
            Min(children) => children[0].eval(ctx).min(children[1].eval(ctx)),
            Max(children) => children[0].eval(ctx).max(children[1].eval(ctx)),
            Const(value) => *value,
            Var(variable) => variable.value(ctx),
        }
    }

    fn grow<T: Variable>(
        generator: &impl RandomNodeGenerator<T>,
        rng: &mut impl Rng,
        depth: usize,
        limit: usize,
    ) -> Node<T> {
        use Node::*;
        if depth >= limit {
            return match generator.generate_no_children(rng) {
                NodeType::Const(value) => Const(value),
                NodeType::Var(variable) => Var(variable),
                _ => panic!(),
            };
        }
        let new_depth = depth + 1;
        let new_node_type = generator.generate(rng);
        let fun = |_| Box::new(Self::grow(generator, rng, new_depth, limit));
        match new_node_type {
            NodeType::If4 => If4(array::from_fn(fun)),
            NodeType::Add => Add(array::from_fn(fun)),
            NodeType::Sub => Sub(array::from_fn(fun)),
            NodeType::Mul => Mul(array::from_fn(fun)),
            NodeType::Div => Div(array::from_fn(fun)),
            NodeType::Neg => Neg(array::from_fn(fun)),
            NodeType::Min => Min(array::from_fn(fun)),
            NodeType::Max => Max(array::from_fn(fun)),
            NodeType::Const(value) => Const(value),
            NodeType::Var(variable) => Var(variable),
        }
    }

    // TODO test that the rust code is identical to eval()
    fn to_rust(&self) -> String {
        use Node::*;
        match self {
            If4([c0, c1, c2, c3]) => {
                format!(
                    "if {} < {} {{ {} }} else {{ {} }}",
                    c0.to_rust(),
                    c1.to_rust(),
                    c2.to_rust(),
                    c3.to_rust()
                )
            }
            Add([c0, c1]) => match (**c0).clone() {
                c0 @ If4(_) => format!("({}) + {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} + {}", c0.to_rust(), c1.to_rust()),
            },
            Sub([c0, c1]) => match (**c0).clone() {
                c0 @ If4(_) => format!("({}) - {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} - {}", c0.to_rust(), c1.to_rust()),
            },
            Mul([c0, c1]) => match (**c0).clone() {
                c0 @ If4(_) => format!("({}) * {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} * {}", c0.to_rust(), c1.to_rust()),
            },
            Div([c0, c1]) => match (**c0).clone() {
                c0 @ If4(_) => format!("({}) / {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} / {}", c0.to_rust(), c1.to_rust()),
            },
            Neg([c0]) => format!("-{}", c0.to_rust()),
            Min([c0, c1]) => format!("f64::min({}, {})", c0.to_rust(), c1.to_rust()),
            Max([c0, c1]) => format!("f64::max({}, {})", c0.to_rust(), c1.to_rust()),
            Const(value) => format!("{value:.2}"),
            Var(v) => v.name().to_string(),
        }
    }

    fn to_dot(&self, id: &mut usize) -> String {
        use Node::*;
        let current_id = *id;
        let mut s = String::new();

        let mut children_to_dot = |name: &str, children: &[Box<Node<V>>]| {
            s.push_str(format!("n{current_id}[label=\"{name}\"]\n").as_str());
            for c in children {
                *id += 1;
                let child_id = *id;
                s.push_str(format!("n{current_id} -> n{child_id}\n").as_str());
                s.push_str(c.to_dot(id).as_str());
            }
        };

        match self {
            If4(children) => children_to_dot("if4", children),
            Add(children) => children_to_dot("add", children),
            Sub(children) => children_to_dot("sub", children),
            Mul(children) => children_to_dot("mul", children),
            Div(children) => children_to_dot("div", children),
            Neg(children) => children_to_dot("neg", children),
            Min(children) => children_to_dot("min", children),
            Max(children) => children_to_dot("max", children),
            Const(value) => s.push_str(format!("n{current_id}[label=\"{value}\"]\n").as_str()),
            Var(v) => s.push_str(format!("n{current_id}[label=\"{}\"]\n", v.name()).as_str()),
        }
        s
    }

    // TODO test that the simplified version is identical to non-simplified
    fn simplify(&mut self) {
        use Node::*;
        for c in self.children_mut().unwrap_or(&mut []) {
            c.simplify();
        }
        *self = match self {
            If4([c0, c1, c2, c3]) => {
                if let (Const(c0), Const(c1)) = (c0.as_ref(), c1.as_ref()) {
                    if c0 < c1 {
                        (**c2).clone()
                    } else {
                        (**c3).clone()
                    }
                } else if *c0 == *c1 || *c2 == *c3 {
                    (**c2).clone()
                } else {
                    self.clone()
                }
            }
            Add([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Const(*c0 + *c1),
                (Const(v), _) if v.is_zero() => (**c1).clone(),
                (_, Const(v)) if v.is_zero() => (**c0).clone(),
                (c0, c1) if c0 == c1 => Mul([
                    Box::new(Const(V::Value::ONE.double())),
                    Box::new(c0.clone()),
                ]),
                (Mul([c00, c01]), c1) if *c1 == **c01 => {
                    let mut inner_add = Add([c00.clone(), Box::new(Const(V::Value::ONE))]);
                    inner_add.simplify();
                    Mul([Box::new(inner_add), c01.clone()])
                }
                // (Add([c0, c1]), c2) | (c2, Add([c0, c1])) => {
                //     let map = [**c0, **c1, *c2]
                //         .iter()
                //         .map(|n| (n.to_discriminant(), n))
                //         .into_group_map();

                //     match map.len() {
                //         2 => (),
                //         1 => Mul([Box::new(Constant(Value(3.0))), map.iter().next().unwrap()]),
                //         other => self.clone(),
                //     }
                // }
                // (Add([c00, c01]), c1) if **c00 == *c1 => Add([
                //     Box::new(Mul([Box::new(Constant(Value(2.0))), Box::new(c1.clone())])),
                //     c01.clone(),
                // ]),
                // (Add([c00, c01]), c1) if **c01 == *c1 => Add([
                //     Box::new(Mul([Box::new(Constant(Value(2.0))), Box::new(c1.clone())])),
                //     c00.clone(),
                // ]),
                // (c0, Add([c10, c11])) if **c11 == *c0 => Add([
                //     Box::new(Mul([Box::new(Constant(Value(2.0))), Box::new(c0.clone())])),
                //     c10.clone(),
                // ]),
                _ => self.clone(),
            },
            Sub([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Const(*c0 - *c1),
                (Const(v), _) if v.is_zero() => (**c1).clone(),
                (_, Const(v)) if v.is_zero() => (**c0).clone(),
                (c0, Neg([c1])) => Add([Box::new(c0.clone()), c1.clone()]),
                (left, right) if left == right => Const(V::Value::ZERO),
                _ => self.clone(),
            },
            Mul([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Const(*c0 * *c1),
                (Const(v), _) | (_, Const(v)) if v.is_zero() => Const(V::Value::ZERO),
                (Const(v), _) if v.is_one() => (**c1).clone(),
                (_, Const(v)) if v.is_one() => (**c0).clone(),
                (c0, c1 @ Const(_)) => Mul([Box::new(c1.clone()), Box::new(c0.clone())]),
                _ => self.clone(),
            },
            Div([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(v), _) | (_, Const(v)) if v.is_zero() => Const(V::Value::ZERO), /* zero division */
                (Const(c0), Const(c1)) => Const(*c0 / *c1),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Neg([c0]) => match c0.as_ref() {
                Const(v) => Const(v.neg()),
                Neg([inner]) => (**inner).clone(), // double negation
                _ => self.clone(),
            },
            Min([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Const(c0.min(*c1)),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Max([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Const(c0.max(*c1)),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Const(_) | Var(_) => self.clone(),
        };
    }
}

impl_value_for_floats!(f32, f64);
impl_value_for_signed_ints!(i8, i16, i32, i64, i128, isize);

#[doc(hidden)]
#[macro_export]
macro_rules! impl_value_for_floats {
    ($($t:ty),+ $(,)?) => {
        $(
            impl Value for $t {
                const MAX:Self = Self::MAX;
                const MIN:Self = Self::MIN;

                fn min(self, other:Self) -> Self {
                    self.min(other)
                }

                fn max(self, other: Self) -> Self {
                    Float::max(self, other)
                }

                fn saturating_add(self, other: Self) -> Self {
                    Self::handle_infinity(self + other)
                }

                fn saturating_sub(self, other: Self) -> Self {
                    Self::handle_infinity(self - other)
                }

                fn saturating_mul(self, other: Self) -> Self {
                    Self::handle_infinity(self * other)
                }

                fn saturating_div(self, other: Self) -> Self {
                    Self::handle_infinity(self / other)
                }

                fn saturating_neg(self) -> Self {
                    Self::handle_infinity(-self)
                }

                fn double(self) -> Self {
                    Self::handle_infinity(self * 2.0 as $t)
                }

                fn is_zero(self) -> bool {
                    self.abs() < Self::EPSILON
                }

                fn handle_infinity(val: Self) -> Self {
                    if val == <$t>::INFINITY {
                        Self::MAX
                    } else if val == <$t>::NEG_INFINITY {
                        Self::MIN
                    } else {
                        val
                    }
                }
            }
        )+
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_value_for_signed_ints {
    ($($t:ty),+ $(,)?) => {
        $(
            impl Value for $t {
                const MAX:Self = Self::MAX;
                const MIN:Self = Self::MIN;

                fn min(self, other:Self) -> Self {
                    Ord::min(self, other)
                }

                fn max(self, other: Self) -> Self {
                    Ord::max(self, other)
                }

                fn saturating_add(self, other: Self) -> Self {
                    <$t>::saturating_add(self, other)
                }

                fn saturating_sub(self, other: Self) -> Self {
                    <$t>::saturating_sub(self, other)
                }

                fn saturating_mul(self, other: Self) -> Self {
                    <$t>::saturating_mul(self, other)
                }

                fn saturating_div(self, other:Self) -> Self {
                    (Saturating(self) / Saturating(other)).0
                }

                fn saturating_neg(self) -> Self {
                    (-Saturating(self)).0
                }

                fn double(self) -> Self {
                    self * 2 as $t
                }

                fn is_zero(self) -> bool {
                    self == Self::ZERO
                }

                // not needed for ints
                fn handle_infinity(val: Self) -> Self { val }
            }
        )+
    };
}

#[cfg(test)]
mod test {

    // #[test]
    // fn testsfd() {
    //     let tree = Tree {
    //         root: Node::Add([
    //             Box::new(Node::Const(0.0.into())),
    //             Box::new(Node::Const(1.0.into())),
    //         ]),
    //     };

    //     let mut rng = Random::seed_from_u64(0);

    //     let tree = Tree::new_random(&mut rng);

    //     println!("{tree:?}");
    // }

    // #[test]
    // fn simple() {
    //     let res = Node::If4([
    //         Box::new(Node::Const(1.0.into())),
    //         Box::new(Node::Const(2.0.into())),
    //         Box::new(Node::Const(0.0.into())),
    //         Box::new(Node::Const(2.0.into())),
    //     ])
    //     .eval(&Context::default());
    //     assert!(res.abs() < f64::EPSILON);
    // }
}

// enum Node {
//     If4([NodeIndex; 4]),

//     Add([NodeIndex; 2]),
//     Sub([NodeIndex; 2]),
//     Div([NodeIndex; 2]),
//     Mul([NodeIndex; 2]),

//     Neg(NodeIndex),
//     Min([NodeIndex; 2]),
//     Max([NodeIndex; 2]),

//     Constant(f64),
// }

// fn test() {
//     // let mut tree = Tree::new(1);
//     // tree.
// }

// #[derive(Copy, Clone, From, Into)]
// struct NodeIndex(u16);

// struct GpTree {
//     nodes: VecMap<NodeIndex, Node>,
// }
