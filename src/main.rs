use std::{f32, fmt::Debug, hash::Hash};

use kiss3d::{
    light::Light,
    nalgebra::{Point3, Translation3},
    window::Window,
};
use num_traits::{FromPrimitive, PrimInt, ToPrimitive, Unsigned};
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform, uniform::SampleUniform},
    rngs::SmallRng,
    seq::SliceRandom,
};

include!(concat!(env!("OUT_DIR"), "/generated_city_data.rs"));

trait CityIdx: PrimInt + Unsigned + Debug + Copy + Ord + Default + FromPrimitive + ToPrimitive {
    #[inline(always)]
    fn as_index(&self) -> usize {
        self.to_usize().unwrap()
    }
}
impl<T: PrimInt + Unsigned + Debug + Copy + Ord + Default + FromPrimitive + ToPrimitive> CityIdx
    for T
{
}

#[derive(Debug, Clone, Copy)]
struct City {
    x: f32, // Performance: consider lower precision (f16)
    y: f32,
}

impl City {
    #[inline(always)]
    fn distance_to(&self, other: &City) -> f32 {
        ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
    }
}

#[derive(Clone, Debug)]
struct Genome<Idx: CityIdx, const N: usize> {
    // Performance: only store city index
    path: [Idx; N],
    fitness: Option<f32>,
}

impl<Idx: CityIdx, const N: usize> Genome<Idx, N> {
    #[inline]
    fn fitness(&mut self, cities: &[City; N]) -> f32 {
        let mut total_distance = 0.0;
        for i in 0..(N - 1) {
            let city1 = &cities[self.path[i].as_index()];
            let city2 = &cities[self.path[i + 1].as_index()];
            total_distance += city1.distance_to(city2);
        }

        let first_city = &cities[self.path[0].as_index()];
        let last_city = &cities[self.path[N - 1].as_index()];
        total_distance += last_city.distance_to(first_city);

        self.fitness = Some(total_distance);
        total_distance
    }
    #[inline]
    fn mutate_genome<R>(&mut self, rng: &mut R, cascade_rate: f64)
    where
        R: Rng,
    {
        let mut first = true;
        while first || rng.random_bool(cascade_rate) {
            first = false;

            let i1 = rng.random_range(0..N);
            let mut i2 = rng.random_range(0..N - 1);

            if i1 == i2 {
                i2 += 1;
            }

            self.path.swap(i1, i2);
        }
    }
}

#[inline]
fn random_genome<Idx: CityIdx + SampleUniform, const N: usize, R>(rng: &mut R) -> Genome<Idx, N>
where
    StandardUniform: Distribution<Idx>,
    R: Rng,
{
    if N == 0 {
        panic!("N cannot be zero");
    }
    let mut indices: [Idx; N] = std::array::from_fn(|i| Idx::from_usize(i).unwrap());
    indices.shuffle(rng);

    Genome {
        path: indices,
        fitness: None,
    }
}

fn breed_ox1<Idx: CityIdx + Hash, const N: usize, R>(
    p1: &Genome<Idx, N>,
    p2: &Genome<Idx, N>,
    rng: &mut R,
) -> (Genome<Idx, N>, Genome<Idx, N>)
where
    R: Rng,
{
    if N < 3 {
        panic!("N must be larger than three for crossover");
    }

    // Consideration: don't hardcode minimum crossover subsequence length
    let length = rng.random_range(1..N - 1);
    let substring_start = rng.random_range(0..(N - length + 1)); // Inclusive
    let substring_end = substring_start + length; // Exclusive

    let mut c1_path: [Idx; N] = [Idx::default(); N];
    let mut c2_path: [Idx; N] = [Idx::default(); N];
    let mut c1_set: [bool; N] = [bool::default(); N];
    let mut c2_set: [bool; N] = [bool::default(); N];
    for i in substring_start..substring_end {
        c1_path[i] = p1.path[i];
        c1_set[p1.path[i].as_index()] = true;
        c2_path[i] = p2.path[i];
        c2_set[p2.path[i].as_index()] = true;
    }

    let mut p2_i = 0;
    let mut c1_i = 0;
    while p2_i < N {
        let p2_city = p2.path[p2_i];
        if c1_i == substring_start {
            c1_i = substring_end;
        }
        if !c1_set[p2_city.as_index()] {
            c1_path[c1_i] = p2_city;
            c1_i += 1;
        }
        p2_i += 1;
    }

    let mut p1_i = 0;
    let mut c2_i = 0;
    while p1_i < N {
        let p1_city = p1.path[p1_i];
        if c2_i == substring_start {
            c2_i = substring_end;
        }
        if !c2_set[p1_city.as_index()] {
            c2_path[c2_i] = p1_city;
            c2_i += 1;
        }
        p1_i += 1;
    }

    (
        Genome {
            path: c1_path,
            fitness: None,
        },
        Genome {
            path: c2_path,
            fitness: None,
        },
    )
}

fn greedy_genome<Idx: CityIdx, const N: usize, R: Rng>(
    cities_data: &[City; N],
    rng: &mut R,
) -> Genome<Idx, N> {
    if N < 2 {
        panic!("N(cities) must be greater than 2.");
    }

    let mut path = [Idx::default(); N];
    let mut visited = vec![false; N];
    let mut current_city_idx_val = Idx::from_usize(rng.random_range(0..N)).unwrap();

    path[0] = current_city_idx_val;
    visited[current_city_idx_val.as_index()] = true;
    let mut path_len = 1;

    while path_len < N {
        let current_city_data = &cities_data[current_city_idx_val.as_index()];
        let mut nearest_neighbor_idx_val = Idx::default();
        let mut min_distance = f32::MAX;

        for i in 0..N {
            let potential_next_city_idx_val = Idx::from_usize(i).unwrap();
            if !visited[potential_next_city_idx_val.as_index()] {
                let distance = current_city_data
                    .distance_to(&cities_data[potential_next_city_idx_val.as_index()]);
                if distance < min_distance {
                    min_distance = distance;
                    nearest_neighbor_idx_val = potential_next_city_idx_val;
                }
            }
        }
        path[path_len] = nearest_neighbor_idx_val;
        visited[nearest_neighbor_idx_val.as_index()] = true;
        current_city_idx_val = nearest_neighbor_idx_val;
        path_len += 1;
    }

    Genome {
        path,
        fitness: None,
    }
}

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const POPULATION_SIZE: usize = 1000;
const NUM_GENERATIONS: usize = 300000;
const MUTATION_RATE: f64 = 0.2;
const CASCADING_MUTATION_RATE: f64 = 0.4;
const CROSSOVER_RATE: f64 = 0.8;
const N_ELITES: usize = POPULATION_SIZE / 25;

const RENDER: bool = false;
const VISUALIZATION_SCALE_DIV: f32 = 2048.0;
const SPHERE_RADIUS: f32 = 5.0 / VISUALIZATION_SCALE_DIV;

const NUM_GREEDY_TO_INJECT: usize = POPULATION_SIZE / 100;
const EXACT_GREEDY_INJECT_GEN: usize = 20000;

fn main() {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEEBABE);
    println!(
        "Program compiled for {} cities from {}.",
        NUM_CITIES,
        CITIES_DATA.len() // CITIES_DATA.len() = NUM_CITIES
    );
    println!(
        "Selected CityIndex type: {}",
        std::any::type_name::<CityIndex>()
    );

    let mut population: Vec<Genome<CityIndex, NUM_CITIES>> = (0..POPULATION_SIZE)
        .map(|_| random_genome::<CityIndex, NUM_CITIES, _>(&mut rng))
        .collect();

    println!(
        "Injecting up to {} greedy genes and their mutants...",
        NUM_GREEDY_TO_INJECT
    );
    let mut min: f32 = f32::MAX;
    let mut max: f32 = f32::MIN;

    let num_to_inject = usize::min(NUM_CITIES, NUM_GREEDY_TO_INJECT);
    (0..num_to_inject).for_each(|i| {
        let mut mutant_gene = greedy_genome(&CITIES_DATA, &mut rng);
        mutant_gene.fitness(&CITIES_DATA);
        println!("pre-retard fitness: {}", &mutant_gene.fitness.unwrap());
        mutant_gene.mutate_genome(&mut rng, 0.3);
        mutant_gene.mutate_genome(&mut rng, 0.3);
        mutant_gene.mutate_genome(&mut rng, 0.3);
        mutant_gene.mutate_genome(&mut rng, 0.3);
        mutant_gene.mutate_genome(&mut rng, 0.3);
        mutant_gene.fitness(&CITIES_DATA);
        let fitness = mutant_gene.fitness(&CITIES_DATA);
        min = f32::min(min, fitness);
        max = f32::max(max, fitness);
        println!("retard fitness: {}", fitness);

        population[i] = mutant_gene;
    });
    println!("Greedy gene (min,max) fitness: ({},{})", min, max);

    let mut best_genome_overall: Option<Genome<CityIndex, NUM_CITIES>> = None;
    for generation in 0..NUM_GENERATIONS {
        for genome in population.iter_mut() {
            if genome.fitness.is_none() {
                genome.fitness(&CITIES_DATA);
            }
        }

        population.sort_unstable_by(|a, b| {
            a.fitness
                .unwrap()
                .partial_cmp(&b.fitness.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if generation == EXACT_GREEDY_INJECT_GEN {
            let mut gg = greedy_genome(&CITIES_DATA, &mut rng);
            println!(
                "inserted greedy chad with {} fitness into a group led by a {} fitness ape",
                gg.fitness(&CITIES_DATA),
                population[0]
                    .fitness
                    .or_else(|| Some(population[0].fitness(&CITIES_DATA)))
                    .unwrap()
            );
            population[POPULATION_SIZE - 1] = gg;
        }

        if best_genome_overall.is_none()
            || population[0].fitness.unwrap()
                < best_genome_overall.as_ref().unwrap().fitness.unwrap()
        {
            best_genome_overall = Some(population[0].clone());
            println!(
                "Generation {}: New best fitness = {}",
                generation,
                best_genome_overall.as_ref().unwrap().fitness.unwrap()
            );
        }

        let mut new_population = Vec::with_capacity(POPULATION_SIZE);
        for i in 0..N_ELITES {
            if i < population.len() {
                new_population.push(population[i].clone());
            }
        }

        while new_population.len() < POPULATION_SIZE {
            let parent1 = &population[rng.random_range(0..POPULATION_SIZE / 2)]; // Select from best half
            let parent2 = &population[rng.random_range(0..POPULATION_SIZE / 2)];

            if rng.random_bool(CROSSOVER_RATE) && NUM_CITIES >= 3 {
                let (mut child1, mut child2) = breed_ox1(parent1, parent2, &mut rng);

                if rng.random_bool(MUTATION_RATE) {
                    child1.mutate_genome(&mut rng, CASCADING_MUTATION_RATE);
                    child1.mutate_genome(&mut rng, CASCADING_MUTATION_RATE);
                }
                if rng.random_bool(MUTATION_RATE) {
                    child2.mutate_genome(&mut rng, CASCADING_MUTATION_RATE);
                    child2.mutate_genome(&mut rng, CASCADING_MUTATION_RATE);
                }
                new_population.push(child1);
                if new_population.len() < POPULATION_SIZE {
                    new_population.push(child2);
                }
            } else {
                new_population.push(parent1.clone());
                if new_population.len() < POPULATION_SIZE {
                    new_population.push(parent2.clone());
                }
            }
        }
        population = new_population;
    }

    if let Some(ref best) = best_genome_overall {
        println!("Finished in {NUM_GENERATIONS} generations!");
        println!("Best fitness found: {}", best.fitness.unwrap());
        println!(
            "Best path: {:?}",
            best.path
                .iter()
                .map(|idx| idx.as_index())
                .collect::<Vec<_>>()
        );
    } else {
        println!("No solution found.");
    }

    if RENDER {
        println!("Initializing visualization...");
        let mut window = Window::new("TSP GA Visualization");
        window.set_light(Light::StickToCamera);

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for city in CITIES_DATA.iter() {
            sum_x += city.x;
            sum_y += city.y;
        }
        let num_cities_f32 = CITIES_DATA.len() as f32;
        let center_x = sum_x / num_cities_f32;
        let center_y = sum_y / num_cities_f32;

        let city_points_3d: Vec<Point3<f32>> = CITIES_DATA
            .iter()
            .map(|city| {
                let x = (city.x - center_x) / VISUALIZATION_SCALE_DIV;
                let y = (city.y - center_y) / VISUALIZATION_SCALE_DIV;
                window
                    .add_sphere(SPHERE_RADIUS)
                    .set_local_translation(Translation3::new(x, y, 0.0));
                Point3::new(x, y, 0.0)
            })
            .collect();

        // port from 2024R
        let line_color_path = Point3::new(0.0, 1.0, 0.0);
        while window.render() {
            if let Some(best) = &best_genome_overall {
                let path_indices = &best.path;
                for i in 0..(NUM_CITIES) {
                    let city1_idx_in_path = path_indices[i].as_index();
                    let city2_idx_in_path = path_indices[(i + 1) % NUM_CITIES].as_index();

                    let p1 = city_points_3d[city1_idx_in_path];
                    let p2 = city_points_3d[city2_idx_in_path];
                    window.draw_line(&p1, &p2, &line_color_path);
                }
            }
        }
        println!("Visualization window closed.");
    }
}
