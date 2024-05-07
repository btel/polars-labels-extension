#![allow(clippy::unused_unit)]
use polars::prelude::arity::binary_elementwise;
use polars::prelude::*;
use pyo3::types::PyIterator;
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
        DataType::List(Box::new(DataType::Int64)),
    );
    Ok(field.clone())
}

#[polars_expr(output_type_func=list_idx_dtype)]
fn to_sparse(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].i64()?;
    let mut cols: Vec<PolarsResult<&ChunkedArray<Int64Type>>> = inputs
        .iter()
        .map(|x| x.i64())
        .collect::<Vec<PolarsResult<&ChunkedArray<Int64Type>>>>();
    let mut cols: Vec<(&str, Box<dyn PolarsIterator<Item = Option<i64>>>)> = cols
        .into_iter()
        .map(|x| {
            let v = x.unwrap();
            (v.name(), v.into_iter())
        })
        .collect();

    let mut builder: ListPrimitiveChunkedBuilder<Int64Type> =
        ListPrimitiveChunkedBuilder::new(left.name(), left.len(), inputs.len(), DataType::Int64);

    loop {
        let values: Option<Vec<Option<i64>>> = cols
            .iter_mut()
            .map(|(name, x)| {
                let value = x.next();
                match value {
                    Some(Some(v)) => {
                        if (v == 1) {
                            Some(Some(name.parse::<i64>().unwrap()))
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
            builder.append_series(&Series::from_vec(
                &"row",
                vec.iter()
                    .filter_map(|x: &Option<i64>| *x)
                    .collect::<Vec<i64>>(),
            ))?;
        } else {
            break;
        }
    }
    let out = builder.finish();
    Ok(out.into_series())
}
