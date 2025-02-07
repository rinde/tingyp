use genetic_progrs::Evaluator;
use genetic_progrs::Evolver;
use genetic_progrs::NodeType;
use genetic_progrs::Tree;
use genetic_progrs::Variable;
use genetic_progrs::WeightedNodeGenerator;

pub fn main() {
    use NodeType::*;
    let generator = WeightedNodeGenerator::new(
        [
            (Add, 1),
            (Sub, 1),
            (Mul, 1),
            (Div, 1),
            (Neg, 1),
            (Const(0.0), 1),
            (Const(1.0), 1),
            (Const(2.0), 1),
            (Const(10.0), 1),
            (Var(Vars::X), 5),
        ],
        [
            (Const(0.0), 1),
            (Const(1.0), 1),
            (Const(2.0), 1),
            (Const(10.0), 1),
            (Var(Vars::X), 1),
        ],
    );

    let mut pop = Evolver::new(500, 6, generator, 0);

    let best = pop.evolve(&Polynomial, 100);

    println!("{best:?}");

    println!("{}", best.0.to_rust());

    let mut simple = best.0.clone();
    simple.simplify();
    println!("----");
    println!("{simple:?}");
    println!("{}", simple.to_rust());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Vars {
    X,
}

#[derive(Default, Clone)]
struct Context {
    x: f64,
}

impl Variable for Vars {
    type Context = Context;
    type Value = f64;

    fn name(&self) -> &'static str {
        match self {
            Self::X => "x",
        }
    }

    fn value(&self, context: &Self::Context) -> f64 {
        match self {
            Self::X => context.x,
        }
    }
}

struct Polynomial;

impl Evaluator<Vars> for Polynomial {
    fn evaluate(&self, _gen: usize, _last_gen: bool, individual: &Tree<Vars>) -> f64 {
        let mut error_sum = 0.0;
        for x in 0..100 {
            let x = x as f64 / 100.0;
            let expected_y = (x * 3.0) + 4.0;
            let context = Context { x: x };
            let actual_y = individual.eval(&context);
            error_sum += (actual_y - expected_y).abs();
        }
        error_sum
    }
}
