#![allow(clippy::unused_unit)]
use polars::prelude::arity::binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(|value: &str, output: &mut String| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}

fn sum_columns(left: Int64Chunked, right: &Series) -> Int64Chunked {
    binary_elementwise(
        &left,
        right.i64().unwrap(),
        |left: Option<i64>, right: Option<i64>| match (left, right) {
            (Some(left), Some(right)) => Some(left + right),
            _ => None,
        },
    )
}

#[polars_expr(output_type=Int64)]
fn sum_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let left: Int64Chunked = inputs[0].i64()?.clone();
    // let right: &Int64Chunked = inputs[1].i64()?;
    // Note: there's a faster way of summing two columns, see
    // section 7.
    let out = inputs[1..].iter().fold(left, sum_columns);
    Ok(out.into_series())
}

fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(
        input_fields[0].name(),
        DataType::List(Box::new(DataType::String)),
    );
    Ok(field.clone())
}

#[polars_expr(output_type_func=list_idx_dtype)]
fn to_sparse(inputs: &[Series]) -> PolarsResult<Series> {
    let left = &inputs[0];
    let mut cols: Vec<(&str, std::vec::IntoIter<Option<bool>>)> = inputs
        .iter()
        .map(|x| {
            let values_iterators = match x.dtype() {
                DataType::Int64 => x
                    .i64()
                    .unwrap()
                    .into_iter()
                    .map(|x| x.map(|v| v == 1))
                    .collect::<Vec<_>>()
                    .into_iter(),
                DataType::Boolean => x
                    .bool()
                    .unwrap()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_iter(),
                _ => panic!("unsupported column data type"),
            };
            (x.name(), values_iterators)
        })
        .collect();

    let mut builder: ListStringChunkedBuilder =
        ListStringChunkedBuilder::new(left.name(), left.len(), inputs.len());

    loop {
        let values: Option<Vec<Option<&str>>> = cols
            .iter_mut()
            .map(|(name, x)| {
                let value = x.next();
                match value {
                    Some(Some(v)) => {
                        if v {
                            Some(Some(*name))
                        } else {
                            Some(None)
                        }
                    }
                    Some(None) => Some(None),
                    None => None,
                }
            })
            .collect();
        if let Some(vec) = values {
            builder.append_values_iter(vec.iter().filter_map(|x: &Option<&str>| *x));
        } else {
            break;
        }
    }
    let out = builder.finish();
    Ok(out.into_series())
}
