//!
use std::array;
use std::fmt::Display;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Neg;

use educe::Educe;
use num::traits::ConstOne;
use num::traits::ConstZero;
use num::Float;
use num::Num;
use num::One;
use num::ToPrimitive;
use num::Zero;
use rand::seq::index;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde::Serialize;
use std::num::Saturating;
use strum::EnumDiscriminants;
use strum::EnumMessage;
use strum::EnumString;
use strum::VariantArray;

pub type Random = Xoshiro256PlusPlus;

pub trait Variable: Clone + Copy + Eq + PartialEq + std::fmt::Debug + Send + Sync {
    type Value: Value + Send + Sync;

    type Context: Send + Sync;

    fn name(&self) -> &'static str;

    fn value(&self, context: &Self::Context) -> Self::Value;
}

pub trait Value:
    Num
    + std::ops::Neg<Output = Self>
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

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn saturating_add(self, other: Self) -> Self;
    fn saturating_sub(self, other: Self) -> Self;
    fn saturating_mul(self, other: Self) -> Self;
    fn saturating_div(self, other: Self) -> Self;
    fn saturating_neg(self) -> Self;
    fn double(self) -> Self;
    fn is_zero(self) -> bool;
    fn handle_infinity(value: Self) -> Self;
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
    pub fn has_children(&self) -> bool {
        use Node::*;
        match self {
            If4(_) | Add(_) | Sub(_) | Mul(_) | Div(_) | Neg(_) | Min(_) | Max(_) => true,
            Const(_) | Var(_) => false,
        }
    }

    pub fn children(&self) -> impl Iterator<Item = &Box<Node<V>>> {
        use Node::*;
        match self {
            If4(children) => children.iter(),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => children.iter(),
            Neg(children) => children.iter(),
            Const(_) | Var(_) => [].iter(),
        }
    }

    pub fn children_mut(&mut self) -> Option<&mut [Box<Node<V>>]> {
        use Node::*;
        match self {
            If4(children) => Some(children),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => Some(children),
            Neg(children) => Some(children),
            Const(_) | Var(_) => None,
        }
    }

    pub fn eval(&self, ctx: &V::Context) -> V::Value {
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
        use NodeType::*;
        if depth >= limit {
            return match generator.generate_no_children(rng) {
                Const(value) => Node::Const(value),
                Var(variable) => Node::Var(variable),
                _ => panic!(),
            };
        }
        let new_depth = depth + 1;
        let new_node_type = generator.generate(rng);
        let fun = |_| Box::new(Self::grow(generator, rng, new_depth, limit));
        match new_node_type {
            If4 => Node::If4(array::from_fn(fun)),
            Add => Node::Add(array::from_fn(fun)),
            Sub => Node::Sub(array::from_fn(fun)),
            Mul => Node::Mul(array::from_fn(fun)),
            Div => Node::Div(array::from_fn(fun)),
            Neg => Node::Neg(array::from_fn(fun)),
            Min => Node::Min(array::from_fn(fun)),
            Max => Node::Max(array::from_fn(fun)),
            Const(value) => Node::Const(value),
            Var(variable) => Node::Var(variable),
        }
    }

    // TODO test that the rust code is identical to eval()
    fn to_rust(&self) -> String {
        match self {
            Node::If4([c0, c1, c2, c3]) => {
                format!(
                    "if {} < {} {{ {} }} else {{ {} }}",
                    c0.to_rust(),
                    c1.to_rust(),
                    c2.to_rust(),
                    c3.to_rust()
                )
            }
            Node::Add([c0, c1]) => match (**c0).clone() {
                c0 @ Node::If4(_) => format!("({}) + {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} + {}", c0.to_rust(), c1.to_rust()),
            },
            Node::Sub([c0, c1]) => match (**c0).clone() {
                c0 @ Node::If4(_) => format!("({}) - {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} - {}", c0.to_rust(), c1.to_rust()),
            },
            Node::Mul([c0, c1]) => match (**c0).clone() {
                c0 @ Node::If4(_) => format!("({}) * {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} * {}", c0.to_rust(), c1.to_rust()),
            },
            Node::Div([c0, c1]) => match (**c0).clone() {
                c0 @ Node::If4(_) => format!("({}) / {}", c0.to_rust(), c1.to_rust()),
                c0 => format!("{} / {}", c0.to_rust(), c1.to_rust()),
            },
            Node::Neg([c0]) => format!("-{}", c0.to_rust()),
            Node::Min([c0, c1]) => format!("f64::min({}, {})", c0.to_rust(), c1.to_rust()),
            Node::Max([c0, c1]) => format!("f64::max({}, {})", c0.to_rust(), c1.to_rust()),
            Node::Const(value) => format!("{:.2}", value),
            Node::Var(v) => v.name().to_string(),
        }
    }

    // TODO test that the simplified version is identical to non-simplified
    fn simplify(&mut self) {
        use Node::*;
        for c in self.children_mut().unwrap_or(&mut []) {
            c.simplify();
        }
        *self = match self {
            Node::If4([c0, c1, c2, c3]) => {
                if let (Node::Const(c0), Node::Const(c1)) = (c0.as_ref(), c1.as_ref()) {
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
            Node::Add([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const(*c0 + *c1),
                (Const(v), _) if v.is_zero() => (**c1).clone(),
                (_, Const(v)) if v.is_zero() => (**c0).clone(),
                (c0, c1) if c0 == c1 => Node::Mul([
                    Box::new(Node::Const(V::Value::ONE.double())),
                    Box::new(c0.clone()),
                ]),
                (Mul([c00, c01]), c1) if *c1 == **c01 => {
                    let mut inner_add = Node::Add([c00.clone(), Box::new(Const(V::Value::ONE))]);
                    inner_add.simplify();
                    Node::Mul([Box::new(inner_add), c01.clone()])
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
            Node::Sub([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const(*c0 - *c1),
                (Const(v), _) if v.is_zero() => (**c1).clone(),
                (_, Const(v)) if v.is_zero() => (**c0).clone(),
                (c0, Neg([c1])) => Add([Box::new(c0.clone()), c1.clone()]),
                (left, right) if left == right => Node::Const(V::Value::ZERO),
                _ => self.clone(),
            },
            Node::Mul([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const(*c0 * *c1),
                (Const(v), _) | (_, Const(v)) if v.is_zero() => Const(V::Value::ZERO),
                (Const(v), _) if v.is_one() => (**c1).clone(),
                (_, Const(v)) if v.is_one() => (**c0).clone(),
                (c0, c1 @ Const(_)) => Node::Mul([Box::new(c1.clone()), Box::new(c0.clone())]),
                _ => self.clone(),
            },
            Node::Div([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(v), _) | (_, Const(v)) if v.is_zero() => Const(V::Value::ZERO), // zero division
                (Const(c0), Const(c1)) => Node::Const(*c0 / *c1),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Neg([c0]) => match c0.as_ref() {
                Const(v) => Const(v.neg()),
                Neg([inner]) => (**inner).clone(), // double negation
                _ => self.clone(),
            },
            Node::Min([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const(c0.min(*c1)),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Max([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const(c0.max(*c1)),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Const(_) | Node::Var(_) => self.clone(),
        };
    }
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
        let chosen = &mut children[rng.gen_range(0..children.len())];
        if rng.gen_bool(0.5) {
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

pub struct WeightedNodeGenerator<const L: usize, const N: usize, V: Variable> {
    all: [(NodeType<V>, u8); L],
    leafs: [(NodeType<V>, u8); N],
}

impl<const L: usize, const N: usize, V: Variable> WeightedNodeGenerator<L, N, V> {
    pub fn new(all: [(NodeType<V>, u8); L], leafs: [(NodeType<V>, u8); N]) -> Self {
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
        self.all[rng.gen_range(0..L)].0
    }

    fn generate_no_children(&self, rng: &mut impl Rng) -> NodeType<V> {
        self.leafs[rng.gen_range(0..N)].0
    }
}

pub struct Evolver<V: Variable, G> {
    population: Vec<Tree<V>>,
    max_tree_depth: usize,
    generator: G,
    rng: Random,
}

impl<V: Variable, G: RandomNodeGenerator<V>> Evolver<V, G> {
    pub fn new(pop_size: usize, max_tree_depth: usize, generator: G, seed: u64) -> Self {
        let mut rng = Random::seed_from_u64(seed);
        Self {
            population: std::iter::repeat_with(|| {
                Tree::new_random(max_tree_depth, &generator, &mut rng)
            })
            .take(pop_size)
            .collect(),
            max_tree_depth,
            generator,
            rng,
        }
    }

    pub fn evolve(
        &mut self,
        evaluator: &impl Evaluator<V>,
        generations: usize,
    ) -> (Tree<V>, V::Value) {
        let mut generation_best = (Tree::default(), V::Value::MAX);

        let last_gen = generations - 1;
        for gen in 0..generations {
            println!("{gen}");

            let fitness = self
                .population
                .par_iter()
                .map(|tree| evaluator.evaluate(gen, gen == last_gen, tree))
                .collect::<Vec<_>>();
            // let depths = self.trees.iter().map(|t| t.depth()).collect::<Vec<_>>();

            println!(
                "fitness min {} avg {}",
                fitness
                    .iter()
                    .min_by(|l, r| l
                        .partial_cmp(r)
                        .unwrap_or_else(|| panic!("{l} {r} comparison failed")))
                    .unwrap(),
                fitness.iter().copied().sum::<V::Value>().to_f64().unwrap() / fitness.len() as f64
            );
            // println!(
            //     "depth min {} avg {}",
            //     depths
            //         .iter()
            //         .min_by(|l, r| l.partial_cmp(r).unwrap())
            //         .unwrap(),
            //     depths.iter().sum::<usize>() as f64 / fitness.len() as f64
            // );

            // println!("{fitness:?}");

            let best_index = fitness
                .iter()
                .enumerate()
                .min_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            // elitism 1
            let mut next_generation = Vec::with_capacity(self.population.capacity());
            next_generation.push(self.population[best_index].clone());

            generation_best = (self.population[best_index].clone(), fitness[best_index]);

            // if fitness[best_index] < all_time_best.1 {
            //     println!(" >>>> found better {}", fitness[best_index]);
            //     all_time_best = (self.population[best_index].clone(), fitness[best_index]);

            //     if fitness[best_index].is_zero() {
            //         println!("Found optimum");
            //         return all_time_best;
            //     }
            // }

            // crossover 90%
            let crossover_target = (0.9 * (self.population.len() as f64)) as usize;
            while next_generation.len() < crossover_target {
                let mut p1 =
                    self.population[Self::select_by_tournament(&fitness, &mut self.rng)].clone();
                let mut p2 =
                    self.population[Self::select_by_tournament(&fitness, &mut self.rng)].clone();
                p1.crossover(&mut p2, &mut self.rng);

                next_generation.extend(
                    [p1, p2]
                        .into_iter()
                        .filter(|p| p.depth() < self.max_tree_depth),
                );
            }

            // mutation 10% (rest)
            while next_generation.len() < self.population.len() {
                let mut mutated =
                    self.population[Self::select_by_tournament(&fitness, &mut self.rng)].clone();
                mutated.mutate(self.max_tree_depth, &self.generator, &mut self.rng);
                next_generation.push(mutated);
            }

            self.population = next_generation;
        }
        generation_best
    }

    fn select_by_tournament(fitness: &[V::Value], rng: &mut impl Rng) -> usize {
        index::sample(rng, fitness.len(), 7)
            .iter()
            .min_by(|&l, &r| fitness[l].partial_cmp(&fitness[r]).unwrap())
            .unwrap()
    }
}

pub trait Evaluator<V: Variable>: Send + Sync {
    fn evaluate(&self, generation: usize, last: bool, individual: &Tree<V>) -> V::Value;
}

fn _test() {
    let x: f64 = 7.0;
    let _result = x + 1.0 * x + f64::max(x + 2.0, 2.0) + 2.0;
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
