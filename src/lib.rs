use std::array;
use std::hash::Hash;

use derive_more::derive::Add;
use derive_more::derive::AddAssign;
use derive_more::derive::Div;
use derive_more::derive::DivAssign;
use derive_more::derive::From;
use derive_more::derive::Into;
use derive_more::derive::Mul;
use derive_more::derive::MulAssign;
use derive_more::derive::Neg;
use derive_more::derive::Sub;
use derive_more::derive::SubAssign;
use educe::Educe;
use rand::seq::index;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use strum::EnumDiscriminants;
use strum::EnumMessage;
use strum::EnumString;
use strum::VariantArray;

pub type Random = Xoshiro256PlusPlus;

pub trait Variable: Clone + Copy + Eq + PartialEq + std::fmt::Debug {
    type Context;

    fn name(&self) -> &'static str;

    fn value(&self, context: &Self::Context) -> Value;
}

#[derive(Debug, EnumDiscriminants, Clone, Eq, PartialEq)]
#[strum_discriminants(derive(EnumString, EnumMessage, VariantArray, Hash))]
enum Node<T: Variable> {
    If4([Box<Node<T>>; 4]),
    Add([Box<Node<T>>; 2]),
    Sub([Box<Node<T>>; 2]),
    Mul([Box<Node<T>>; 2]),
    Div([Box<Node<T>>; 2]),

    Neg([Box<Node<T>>; 1]),
    Min([Box<Node<T>>; 2]),
    Max([Box<Node<T>>; 2]),
    Const(Value),
    Var(T),
}

// TODO make generic
#[derive(
    Clone,
    Copy,
    Debug,
    From,
    Into,
    Default,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    Neg,
)]
pub struct Value(pub f64);

impl Value {
    pub fn abs(&self) -> Value {
        Self(self.0.abs())
    }
}

impl Eq for Value {}
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Self::partial_cmp(&self, other).unwrap()
    }
}
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: Variable> Node<T> {
    pub fn has_children(&self) -> bool {
        use Node::*;
        match self {
            If4(_) | Add(_) | Sub(_) | Mul(_) | Div(_) | Neg(_) | Min(_) | Max(_) => true,
            Const(_) | Var(_) => false,
        }
    }

    pub fn children(&self) -> impl Iterator<Item = &Box<Node<T>>> {
        use Node::*;
        match self {
            If4(children) => children.iter(),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => children.iter(),
            Neg(children) => children.iter(),
            Const(_) | Var(_) => [].iter(),
        }
    }

    pub fn children_mut(&mut self) -> Option<&mut [Box<Node<T>>]> {
        use Node::*;
        match self {
            If4(children) => Some(children),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => Some(children),
            Neg(children) => Some(children),
            Const(_) | Var(_) => None,
        }
    }

    pub fn eval(&self, ctx: &T::Context) -> Value {
        use Node::*;
        match self {
            If4(children) => {
                if children[0].eval(ctx) < children[1].eval(ctx) {
                    children[2].eval(ctx)
                } else {
                    children[3].eval(ctx)
                }
            }
            Add(children) => children[0].eval(ctx) + children[1].eval(ctx),
            Sub(children) => children[0].eval(ctx) - children[1].eval(ctx),
            Mul(children) => children[0].eval(ctx) * children[1].eval(ctx).0,
            Div(children) => {
                let v1 = children[1].eval(ctx);
                if v1.abs() < Value(f64::EPSILON) {
                    Value(0.0)
                } else {
                    Value(children[0].eval(ctx).0 / v1.0)
                }
            }
            Neg(children) => -children[0].eval(ctx),
            Min(children) => children[0].eval(ctx).min(children[1].eval(ctx)),
            Max(children) => children[0].eval(ctx).max(children[1].eval(ctx)),
            Const(value) => (*value).into(),
            Var(variable) => variable.value(ctx),
        }
    }

    fn grow<V: Variable>(
        generator: &impl RandomNodeGenerator<V>,
        rng: &mut impl Rng,
        depth: usize,
        limit: usize,
    ) -> Node<V> {
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
            Node::Const(value) => format!("{:.2}", value.0),
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
                (Const(c0), Const(c1)) => Node::Const((c0.0 + c1.0).into()),
                (Const(Value(0.0)), _) => (**c1).clone(),
                (_, Const(Value(0.0))) => (**c0).clone(),
                (c0, c1) if c0 == c1 => {
                    Node::Mul([Box::new(Node::Const(Value(2.0))), Box::new(c0.clone())])
                }
                (Mul([c00, c01]), c1) if *c1 == **c01 => {
                    let mut inner_add = Node::Add([c00.clone(), Box::new(Const(Value(1.0)))]);
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
                (Const(c0), Const(c1)) => Node::Const((c0.0 - c1.0).into()),
                (Const(Value(0.0)), _) => (**c1).clone(),
                (_, Const(Value(0.0))) => (**c0).clone(),
                (c0, Neg([c1])) => Add([Box::new(c0.clone()), c1.clone()]),
                (left, right) if left == right => Node::Const(Value(0.0)),
                _ => self.clone(),
            },
            Node::Mul([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const((c0.0 * c1.0).into()),
                (Const(Value(0.0)), _) | (_, Const(Value(0.0))) => Node::Const(Value(0.0)),
                (Const(Value(1.0)), _) => (**c1).clone(),
                (_, Const(Value(1.0))) => (**c0).clone(),
                (c0, c1 @ Const(_)) => Node::Mul([Box::new(c1.clone()), Box::new(c0.clone())]),
                _ => self.clone(),
            },
            Node::Div([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(Value(0.0)), _) | (_, Const(Value(0.0))) => Node::Const(Value(0.0)), // zero division
                (Const(c0), Const(c1)) => Node::Const((c0.0 / c1.0).into()),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Neg([c0]) => match c0.as_ref() {
                Const(v) => Const(Value(-v.0)),
                Neg([inner]) => (**inner).clone(), // double negation
                _ => self.clone(),
            },
            Node::Min([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const(c0.0.min(c1.0).into()),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Max([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Const(c0), Const(c1)) => Node::Const(c0.0.max(c1.0).into()),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Const(_) | Node::Var(_) => self.clone(),
        };
    }
}

#[derive(Debug, Clone, Educe)]
#[educe(Default)]
pub struct Tree<V: Variable> {
    #[educe(Default = Node::Const(0.0.into()))]
    root: Node<V>,
}

impl<V: Variable> Tree<V> {
    const MAX_DEPTH: usize = 16;

    fn new_random(
        max_depth: usize,
        generator: &impl RandomNodeGenerator<V>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            root: Node::<V>::grow(generator, rng, 0, max_depth),
        }
    }

    fn mutate(&mut self, generator: &impl RandomNodeGenerator<V>, rng: &mut impl Rng) {
        let (node, depth) = Self::pick_random(&mut self.root, 0, rng);
        *node = Node::<V>::grow(generator, rng, 0, Self::MAX_DEPTH - depth);
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

    pub fn eval(&self, ctx: &V::Context) -> Value {
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
    Const(Value),
    Var(V),
}

pub trait RandomNodeGenerator<V: Variable> {
    fn generate(&self, rng: &mut impl Rng) -> NodeType<V>;
    fn generate_no_children(&self, rng: &mut impl Rng) -> NodeType<V>;
}

pub struct WeightedNodeGenerator<const L: usize, const N: usize, V: Variable> {
    all: [(NodeType<V>, u8); L],
    no_kids: [(NodeType<V>, u8); N],
}

impl<const L: usize, const N: usize, V: Variable> WeightedNodeGenerator<L, N, V> {
    pub fn new(all: [(NodeType<V>, u8); L], no_kids: [(NodeType<V>, u8); N]) -> Self {
        for (n, _) in no_kids {
            assert!(matches!(n, NodeType::Const(_) | NodeType::Var(_)));
        }
        Self { all, no_kids }
    }
}

impl<const L: usize, const N: usize, V: Variable> RandomNodeGenerator<V>
    for WeightedNodeGenerator<L, N, V>
{
    fn generate(&self, rng: &mut impl Rng) -> NodeType<V> {
        self.all[rng.gen_range(0..L)].0
    }

    fn generate_no_children(&self, rng: &mut impl Rng) -> NodeType<V> {
        self.no_kids[rng.gen_range(0..N)].0
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

    pub fn evolve(&mut self, evaluator: &impl Evaluator<V>, generations: usize) -> (Tree<V>, f64) {
        let mut all_time_best = (Tree::default(), f64::MAX);

        for gen in 0..generations {
            println!("{gen}");

            let fitness = self
                .population
                .iter()
                .map(|tree| evaluator.evaluate(tree))
                .collect::<Vec<_>>();
            // let depths = self.trees.iter().map(|t| t.depth()).collect::<Vec<_>>();

            println!(
                "fitness min {} avg {}",
                fitness
                    .iter()
                    .min_by(|l, r| l.partial_cmp(r).unwrap())
                    .unwrap(),
                fitness.iter().sum::<f64>() / fitness.len() as f64
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
            if fitness[best_index] < all_time_best.1 {
                println!(" >>>> found better {}", fitness[best_index]);
                all_time_best = (self.population[best_index].clone(), fitness[best_index]);

                if fitness[best_index].abs() < f64::EPSILON {
                    println!("Found optimum");
                    return all_time_best;
                }
            }

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
                mutated.mutate(&self.generator, &mut self.rng);
                next_generation.push(mutated);
            }

            self.population = next_generation;
        }
        all_time_best
    }

    fn select_by_tournament(fitness: &[f64], rng: &mut impl Rng) -> usize {
        index::sample(rng, fitness.len(), 7)
            .iter()
            .min_by(|&l, &r| fitness[l].partial_cmp(&fitness[r]).unwrap())
            .unwrap()
    }
}

pub trait Evaluator<V: Variable> {
    fn evaluate(&self, individual: &Tree<V>) -> f64;
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
