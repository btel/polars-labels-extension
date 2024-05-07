#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use polars::prelude::arity::binary_elementwise;
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


fn sum_columns(left: Int64Chunked, right: &Series) -> Int64Chunked
{
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
    let field = Field::new(input_fields[0].name(), DataType::List(Box::new(DataType::Int64)));
    Ok(field.clone())
}

#[polars_expr(output_type_func=list_idx_dtype)]
fn to_sparse(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].i64()?;
    // let right: &Int64Chunked = inputs[1].i64()?;
    // Note: there's a faster way of summing two columns, see
    // section 7.
    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(left.name(), left.len(), inputs.len(), DataType::Int64);

    left.for_each(|x| builder.append_series(&Series::new(&"empty", [x, x])).unwrap());
    let out = builder.finish();
    Ok(out.into_series())
}
