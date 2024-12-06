use std::array;
use std::hash::Hash;

use derive_more::derive::From;
use derive_more::derive::Into;
use itertools::Itertools;
use rand::seq::index;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use strum::EnumDiscriminants;
use strum::EnumMessage;
use strum::EnumString;
use strum::VariantArray;

pub type Random = Xoshiro256PlusPlus;

#[derive(Debug, EnumDiscriminants, Clone, Default, Eq, PartialEq)]
#[strum_discriminants(derive(EnumString, EnumMessage, VariantArray, Hash))]
enum Node {
    If4([Box<Node>; 4]),
    Add([Box<Node>; 2]),
    Sub([Box<Node>; 2]),
    Mul([Box<Node>; 2]),
    Div([Box<Node>; 2]),

    Neg([Box<Node>; 1]),
    Min([Box<Node>; 2]),
    Max([Box<Node>; 2]),
    Constant(Value),
    #[default]
    X,
}

#[derive(Clone, Copy, Debug, From, Into)]
struct Value(f64);

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

impl Node {
    pub fn has_children(&self) -> bool {
        use Node::*;
        match self {
            If4(_) | Add(_) | Sub(_) | Mul(_) | Div(_) | Neg(_) | Min(_) | Max(_) => true,
            Constant(_) | X => false,
        }
    }

    pub fn children(&self) -> impl Iterator<Item = &Box<Node>> {
        use Node::*;
        match self {
            If4(children) => children.iter(),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => children.iter(),
            Neg(children) => children.iter(),
            Constant(_) | X => [].iter(),
        }
    }

    pub fn children_mut(&mut self) -> Option<&mut [Box<Node>]> {
        use Node::*;
        match self {
            If4(children) => Some(children),
            Add(children) | Sub(children) | Mul(children) | Div(children) | Min(children)
            | Max(children) => Some(children),
            Neg(children) => Some(children),
            Constant(_) | X => None,
        }
    }

    pub fn eval(&self, ctx: &Context) -> f64 {
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
            Mul(children) => children[0].eval(ctx) * children[1].eval(ctx),
            Div(children) => {
                let v1 = children[1].eval(ctx);
                if v1.abs() < f64::EPSILON {
                    0.0
                } else {
                    children[0].eval(ctx) / v1
                }
            }
            Neg(children) => -children[0].eval(ctx),
            Min(children) => children[0].eval(ctx).min(children[1].eval(ctx)),
            Max(children) => children[0].eval(ctx).max(children[1].eval(ctx)),
            Constant(value) => (*value).into(),
            X => ctx.x,
        }
    }

    fn random_constant(rng: &mut impl Rng) -> Node {
        [
            Node::Constant(0.0.into()),
            Node::Constant(1.0.into()),
            Node::Constant(2.0.into()),
            Node::Constant(10.0.into()),
        ]
        .choose(rng)
        .unwrap()
        .clone()
    }

    fn grow(rng: &mut impl Rng, depth: usize, limit: usize) -> Self {
        use NodeDiscriminants::*;
        const SELECTED: [NodeDiscriminants; 7] = [
            /*If4,*/ Add, Sub, Mul, Div, Neg, /*Min, Max,*/ Constant, X,
        ];

        if depth >= limit {
            return Self::random_constant(rng);
        }
        let chosen = if rng.gen_bool(0.2) {
            Constant
        } else {
            *SELECTED.choose(rng).unwrap()
        };
        let new_depth = depth + 1;
        let fun = |_| Box::new(Self::grow(rng, new_depth, limit));
        match chosen {
            If4 => Node::If4(array::from_fn(fun)),
            Add => Node::Add(array::from_fn(fun)),
            Sub => Node::Sub(array::from_fn(fun)),
            Mul => Node::Mul(array::from_fn(fun)),
            Div => Node::Div(array::from_fn(fun)),
            Neg => Node::Neg(array::from_fn(fun)),
            Min => Node::Min(array::from_fn(fun)),
            Max => Node::Max(array::from_fn(fun)),
            Constant => Self::random_constant(rng),
            X => Node::X,
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
            Node::Constant(value) => format!("{:.2}", value.0),
            Node::X => "x".to_string(),
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
                if let (Node::Constant(c0), Node::Constant(c1)) = (c0.as_ref(), c1.as_ref()) {
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
                (Constant(c0), Constant(c1)) => Node::Constant((c0.0 + c1.0).into()),
                (Constant(Value(0.0)), _) => (**c1).clone(),
                (_, Constant(Value(0.0))) => (**c0).clone(),
                (c0, c1) if c0 == c1 => {
                    Node::Mul([Box::new(Node::Constant(Value(2.0))), Box::new(c0.clone())])
                }
                (Mul([c00, c01]), c1) if *c1 == **c01 => {
                    let mut inner_add = Node::Add([c00.clone(), Box::new(Constant(Value(1.0)))]);
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
                (Constant(c0), Constant(c1)) => Node::Constant((c0.0 - c1.0).into()),
                (Constant(Value(0.0)), _) => (**c1).clone(),
                (_, Constant(Value(0.0))) => (**c0).clone(),
                (c0, Neg([c1])) => Add([Box::new(c0.clone()), c1.clone()]),
                (left, right) if left == right => Node::Constant(Value(0.0)),
                _ => self.clone(),
            },
            Node::Mul([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Constant(c0), Constant(c1)) => Node::Constant((c0.0 * c1.0).into()),
                (Constant(Value(0.0)), _) | (_, Constant(Value(0.0))) => Node::Constant(Value(0.0)),
                (Constant(Value(1.0)), _) => (**c1).clone(),
                (_, Constant(Value(1.0))) => (**c0).clone(),
                (c0, c1 @ Constant(_)) => Node::Mul([Box::new(c1.clone()), Box::new(c0.clone())]),
                _ => self.clone(),
            },
            Node::Div([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Constant(Value(0.0)), _) | (_, Constant(Value(0.0))) => Node::Constant(Value(0.0)), // zero division
                (Constant(c0), Constant(c1)) => Node::Constant((c0.0 / c1.0).into()),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Neg([c0]) => match c0.as_ref() {
                Constant(v) => Constant(Value(-v.0)),
                Neg([inner]) => (**inner).clone(), // double negation
                _ => self.clone(),
            },
            Node::Min([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Constant(c0), Constant(c1)) => Node::Constant(c0.0.min(c1.0).into()),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Max([c0, c1]) => match (c0.as_ref(), c1.as_ref()) {
                (Constant(c0), Constant(c1)) => Node::Constant(c0.0.max(c1.0).into()),
                (left, right) if left == right => left.clone(),
                _ => self.clone(),
            },
            Node::Constant(_) | Node::X => self.clone(),
        };
    }

    fn to_discriminant(&self) -> NodeDiscriminants {
        use NodeDiscriminants::*;
        match self {
            Node::If4(_) => If4,
            Node::Add(_) => Add,
            Node::Sub(_) => Sub,
            Node::Mul(_) => Mul,
            Node::Div(_) => Div,
            Node::Neg(_) => Neg,
            Node::Min(_) => Min,
            Node::Max(_) => Max,
            Node::Constant(_) => Constant,
            Node::X => X,
        }
    }
}

#[derive(Default, Clone)]
struct Context {
    x: f64,
}

#[derive(Debug, Clone, Default)]
struct Tree {
    root: Node,
}

impl Tree {
    const MAX_DEPTH: usize = 16;

    fn new_random(rng: &mut impl Rng) -> Self {
        Self {
            root: Node::grow(rng, 0, Self::MAX_DEPTH),
        }
    }

    fn mutate(&mut self, rng: &mut impl Rng) {
        let (node, depth) = Self::pick_random(&mut self.root, 0, rng);
        *node = Node::grow(rng, 0, Self::MAX_DEPTH - depth);
    }

    fn crossover(&mut self, other: &mut Self, rng: &mut impl Rng) {
        let (one, _) = Tree::pick_random(&mut self.root, 0, rng);
        let (two, _) = Tree::pick_random(&mut other.root, 0, rng);
        std::mem::swap(one, two);
    }

    fn pick_random<'a>(
        node: &'a mut Node,
        depth: usize,
        rng: &mut impl Rng,
    ) -> (&'a mut Node, usize) {
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

    fn eval(&self, ctx: &Context) -> f64 {
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

    fn to_rust(&self) -> String {
        format!("let result = {};", self.root.to_rust())
    }

    fn simplify(&mut self) {
        self.root.simplify();
    }
}

struct Population {
    trees: Vec<Tree>,
}

impl Population {
    fn new(size: usize, rng: &mut impl Rng) -> Self {
        Self {
            trees: std::iter::repeat_with(|| Tree::new_random(rng))
                .take(size)
                .collect(),
        }
    }

    // calculate fitness
    // save best
    // select

    fn evolve(
        &mut self,
        evaluator: &impl Evaluator,
        generations: usize,
        rng: &mut impl Rng,
    ) -> (Tree, f64) {
        let mut all_time_best = (Tree::default(), f64::MAX);

        for gen in 0..generations {
            println!("{gen}");

            let fitness = self
                .trees
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
            let mut next_generation = Vec::with_capacity(self.trees.capacity());
            next_generation.push(self.trees[best_index].clone());
            if fitness[best_index] < all_time_best.1 {
                println!(" >>>> found better {}", fitness[best_index]);
                all_time_best = (self.trees[best_index].clone(), fitness[best_index]);

                if fitness[best_index].abs() < f64::EPSILON {
                    println!("Found optimum");
                    return all_time_best;
                }
            }

            // crossover 90%
            let crossover_target = (0.9 * (self.trees.len() as f64)) as usize;
            while next_generation.len() < crossover_target {
                let mut p1 = self.trees[Self::select_by_tournament(&fitness, rng)].clone();
                let mut p2 = self.trees[Self::select_by_tournament(&fitness, rng)].clone();
                p1.crossover(&mut p2, rng);

                next_generation
                    .extend([p1, p2].into_iter().filter(|p| p.depth() < Tree::MAX_DEPTH));
            }

            // mutation 10% (rest)
            while next_generation.len() < self.trees.len() {
                let mut mutated = self.trees[Self::select_by_tournament(&fitness, rng)].clone();
                mutated.mutate(rng);
                next_generation.push(mutated);
            }

            self.trees = next_generation;
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

trait Evaluator {
    fn evaluate(&self, individual: &Tree) -> f64;
}

pub fn main() {
    let mut rng = Random::seed_from_u64(0);
    let mut pop = Population::new(500, &mut rng);

    let best = pop.evolve(&Quadratic, 100, &mut rng);

    println!("{best:?}");

    println!("{}", best.0.to_rust());

    let mut simple = best.0.clone();
    simple.simplify();
    println!("----");
    println!("{simple:?}");
    println!("{}", simple.to_rust());
}

fn _test() {
    let x: f64 = 7.0;
    let _result = x + 1.0 * x + f64::max(x + 2.0, 2.0) + 2.0;
    let result = if 10.00
        < 2.00
            + 3.00 * x
                / if if 2.00 + -80.00 * x - 2.00 + -x < 2.00 {
                    x
                } else {
                    if x - if -8.00 < x { 3.00 } else { 1.00 } + -109.00 < 0.00 {
                        1.00
                    } else {
                        x - -10.00
                    }
                } < x
                {
                    10.00
                } else {
                    1.00
                }
            + 2.00
    {
        2.00 + 3.00 * x
            / if if 2.00 + -80.00 * x - 2.00 + -x < 2.00 {
                x
            } else {
                if x - if -8.00 < x { 3.00 } else { 1.00 } + -109.00 < 0.00 {
                    1.00
                } else {
                    x - -10.00
                }
            } < x
            {
                10.00
            } else {
                1.00
            }
            + 2.00
    } else {
        3.00 * x
            / if if 2.00 + -80.00 * x - 2.00 + -x < 2.00 {
                x
            } else {
                if x - if -8.00 < x { 3.00 } else { 1.00 } + -109.00 < 0.00 {
                    1.00
                } else {
                    x - -10.00
                }
            } < x
            {
                10.00
            } else {
                1.00
            }
            + 4.00
    };
}

struct Quadratic;

impl Evaluator for Quadratic {
    fn evaluate(&self, individual: &Tree) -> f64 {
        let mut error_sum = 0.0;
        for x in 0..100 {
            let x = x as f64 / 100.0;
            let expected_y = (x * 2.0) + x + 4.0;
            let actual_y = individual.eval(&Context { x });
            error_sum += (actual_y - expected_y).abs();
        }
        error_sum
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use rand::SeedableRng;

    #[test]
    fn testsfd() {
        let tree = Tree {
            root: Node::Add([
                Box::new(Node::Constant(0.0.into())),
                Box::new(Node::Constant(1.0.into())),
            ]),
        };

        let mut rng = Random::seed_from_u64(0);

        let tree = Tree::new_random(&mut rng);

        println!("{tree:?}");
    }

    #[test]
    fn simple() {
        let res = Node::If4([
            Box::new(Node::Constant(1.0.into())),
            Box::new(Node::Constant(2.0.into())),
            Box::new(Node::Constant(0.0.into())),
            Box::new(Node::Constant(2.0.into())),
        ])
        .eval(&Context::default());
        assert!(res.abs() < f64::EPSILON);
    }
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
