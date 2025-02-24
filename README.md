# Tiny GP

<!-- cargo-rdme start -->

A tiny genetic programming library.

## Example

```rust
use nonzero_lit::usize;
use tinygp::NodeType::*;
use tinygp::*;
let generator = WeightedNodeGenerator::new(
    [(Add, 1), (Sub, 1), (Const(1.0), 1), (Var(Vars::X), 2)],
    [(Const(1.0), 1), (Var(Vars::X), 1)],
);

let population_size = usize!(200);
let max_tree_depth = usize!(6);
let random_seed = 0;
let mut evolver = Evolver::new(population_size, max_tree_depth, generator, random_seed);

let generations = 10;
let (best, best_fitness) = evolver.evolve(&Sub3, generations);

assert!(
    best_fitness <= f64::EPSILON,
    "A perfect program was not found, best fitness {}",
    best_fitness
);

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

/// An evaluator that calculates how closely a Tree matches the function `x - 3`.
struct Sub3;

impl Evaluator<Vars> for Sub3 {
    fn evaluate(&self, _gen: usize, _last_gen: bool, individual: &Tree<Vars>) -> f64 {
        let mut error_sum = 0.0;
        for x in 0..100 {
            let x = f64::from(x);
            let expected_y = x - 3.0;
            let context = Context { x };
            let actual_y = individual.eval(&context);
            error_sum += (actual_y - expected_y).abs();
        }
        error_sum
    }
}
```

<!-- cargo-rdme end -->
