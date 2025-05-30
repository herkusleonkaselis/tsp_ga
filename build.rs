use std::env;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum SmallestViableIndexType {
    TypeU8,
    TypeU16,
    TypeU32,
    TypeU64,
}

#[derive(Debug, Clone, Copy)]
struct BuildCity {
    x: f32,
    y: f32,
}

fn parse_tsp_file(file_path_str: &str) -> io::Result<(usize, Vec<BuildCity>)> {
    let file = File::open(file_path_str)?;
    let reader = io::BufReader::new(file);

    let mut dimension: Option<usize> = None;
    let mut cities = Vec::new();
    let mut in_node_coord_section = false;

    for line_result in reader.lines() {
        let line = line_result?.trim().to_string();

        if line.starts_with("DIMENSION") {
            if let Some(val_str) = line.split(':').nth(1) {
                dimension = Some(val_str.trim().parse().expect("Failed to parse DIMENSION"));
            }
        } else if line == "NODE_COORD_SECTION" {
            in_node_coord_section = true;
            continue;
        } else if line == "EOF" || line.is_empty() {
            if in_node_coord_section {
                break;
            }
            continue;
        }

        if in_node_coord_section {
            let parts: Vec<&str> = line.split_whitespace().collect(); //t?
            if parts.len() >= 3 {
                let x: f32 = parts[1].parse().expect("Invalid city X");
                let y: f32 = parts[2].parse().expect("Invalid city Y");
                cities.push(BuildCity { /*id,*/ x, y }); // ???
            }
        }
    }

    let dim = dimension.expect("DIMENSION not found in TSP file");
    if cities.len() != dim {
        panic!(
            "Mismatch between DIMENSION ({}) and number of cities found ({})",
            dim,
            cities.len()
        );
    }
    Ok((dim, cities))
}

impl SmallestViableIndexType {
    pub fn to_rust_type_str(self) -> &'static str {
        match self {
            SmallestViableIndexType::TypeU8 => "u8",
            SmallestViableIndexType::TypeU16 => "u16",
            SmallestViableIndexType::TypeU32 => "u32",
            SmallestViableIndexType::TypeU64 => "u64",
        }
    }
}

pub fn get_smallest_viable_index_type_from_len(length: usize) -> SmallestViableIndexType {
    if length == 0 {
        return SmallestViableIndexType::TypeU8;
    }
    let max_index = length - 1;
    if max_index <= u8::MAX as usize {
        SmallestViableIndexType::TypeU8
    } else if max_index <= u16::MAX as usize {
        SmallestViableIndexType::TypeU16
    } else if max_index <= u32::MAX as usize {
        SmallestViableIndexType::TypeU32
    } else {
        SmallestViableIndexType::TypeU64
    }
}

fn main() -> io::Result<()> {
    let tsp_file_path = "./lu980.tsp";

    println!("cargo:rerun-if-changed={}", tsp_file_path);
    println!("cargo:rerun-if-changed=build.rs");

    let (num_cities, cities_data) =
        parse_tsp_file(tsp_file_path).expect("Failed to parse TSP file in build.rs");

    let city_index_type = get_smallest_viable_index_type_from_len(num_cities);

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated_city_data.rs");
    let mut f = File::create(dest_path)?;

    writeln!(f, "const NUM_CITIES: usize = {};", num_cities)?;
    writeln!(
        f,
        "pub type CityIndex = {};",
        city_index_type.to_rust_type_str()
    )?;
    writeln!(f, "")?;
    writeln!(f, "pub static CITIES_DATA: [City; NUM_CITIES] = [")?;
    for city in cities_data {
        writeln!(f, "    City {{ x: {:.5e}, y: {:.5e} }},", city.x, city.y)?;
    }
    writeln!(f, "];")?;

    Ok(())
}
