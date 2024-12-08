use genetic_progrs::Evaluator;
use genetic_progrs::Evolver;
use genetic_progrs::NodeType;
use genetic_progrs::Tree;
use genetic_progrs::Value;
use genetic_progrs::Variable;
use genetic_progrs::WeightedNodeGenerator;

pub fn main() {
    let generator = WeightedNodeGenerator::new(
        [
            (NodeType::Add, 1),
            (NodeType::Sub, 1),
            (NodeType::Mul, 1),
            (NodeType::Div, 1),
            (NodeType::Neg, 1),
            (NodeType::Const(Value(0.0)), 1),
            (NodeType::Const(Value(1.0)), 1),
            (NodeType::Const(Value(2.0)), 1),
            (NodeType::Const(Value(10.0)), 1),
            (NodeType::Var(Vars::X), 5),
        ],
        [
            (NodeType::Const(Value(0.0)), 1),
            (NodeType::Const(Value(1.0)), 1),
            (NodeType::Const(Value(2.0)), 1),
            (NodeType::Const(Value(10.0)), 1),
            (NodeType::Var(Vars::X), 1),
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
    x: Value,
}

impl Variable for Vars {
    type Context = Context;

    fn name(&self) -> &'static str {
        match self {
            Self::X => "x",
        }
    }

    fn value(&self, context: &Self::Context) -> Value {
        match self {
            Self::X => context.x,
        }
    }
}

struct Polynomial;

impl<V: Variable<Context = Context>> Evaluator<V> for Polynomial {
    fn evaluate(&self, individual: &Tree<V>) -> f64 {
        let mut error_sum = 0.0;
        for x in 0..100 {
            let x = x as f64 / 100.0;
            let expected_y = Value((x * 2.0) + x + 4.0);
            let context = Context { x: Value(x) };
            let actual_y = individual.eval(&context);
            error_sum += (actual_y - expected_y).abs().0;
        }
        error_sum
    }
}
