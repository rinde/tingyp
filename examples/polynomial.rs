use nonzero_lit::usize;
use tinygp::Evaluator;
use tinygp::Evolver;
use tinygp::NodeType;
use tinygp::Tree;
use tinygp::Variable;
use tinygp::WeightedNodeGenerator;

#[expect(clippy::enum_glob_use, reason = "conciseness")]
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

    let mut pop = Evolver::new(usize!(500), usize!(6), generator, 0);

    let best = pop.evolve(&Polynomial, 100);

    println!("{best:?}");
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
            let x = f64::from(x) / 100.0;
            let expected_y = (x * 3.0) + 4.0;
            let context = Context { x };
            let actual_y = individual.eval(&context);
            error_sum += (actual_y - expected_y).abs();
        }
        error_sum
    }
}
