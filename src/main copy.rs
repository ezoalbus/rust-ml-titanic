use linfa::prelude::*;
// use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
// use linfa::{Float, Label};
use linfa_bayes::GaussianNb;
use linfa_bayes::Result as NBResult;
use polars::prelude::*;
use polars::prelude::Result as PolarResult;
// use ndarray::prelude::*;
// use ndarray::{Array1, ArrayBase, ArrayView2, Axis, Data, Ix2};
// use ndarray::array;
// use polars_core::prelude::*;
// use polars_io::prelude::*;
use csv::{ReaderBuilder, WriterBuilder};
use std::error::Error;
use ndarray_csv::{Array2Reader, Array2Writer};
use std::fs::File;
// use std::path::Path;


fn read_csv2df(path: &str) -> PolarResult<DataFrame> {
    // csvを読み込んで、DataFrameでreturn
    CsvReader::from_path(path)?
            .has_header(true)
            .finish()
}

fn split_x_y(df: &DataFrame) -> (PolarResult<DataFrame>, PolarResult<DataFrame>) {
    // 特徴量とターゲットに分割
    let target = df.select(vec!["Survived"]);
    // let features = df.drop("Survived");
    let features = df.select(vec!["Fare"]);
    (features, target)
}

fn main() -> NBResult<()> {
    //  trainの読み込み
    let train_path = "./data/train.csv";
    let test_path = "./data/test.csv";

    let train_df = read_csv2df(&train_path).unwrap();
    let (train_x, train_y) = split_x_y(&train_df);
    let test_x = read_csv2df(&test_path);
    let test_x = test_x.unwrap().select(vec!["Fare"]);
    // let val_df = read_csv2df(&val_path); 
    // println!("{:?}", train_x);


    // // データセットを読み込み、ターゲット（ワインの品質の評価値）を二値に変換
    // //     品質は、0から10の間で評価されており、0が最低10が最高
    // let (train, valid) = linfa_datasets::winequality()
    //     .map_targets(|x| if *x > 6 { "good" } else { "bad" })
    //     .split_with_ratio(0.9);

    // モデルの訓練
    // 欠測値の削除
    // let train_x = train_x.unwrap().drop_nulls(None).unwrap();
    let train_x = train_x.unwrap().to_ndarray::<Float64Type>().unwrap();
    let train_y = train_y.unwrap().to_ndarray::<Int64Type>().unwrap();
    // let train_y = Array::from_iter(train_y.iter()).map(|x| if *x > 0 {"Survived"} else {"Not Survived"});
    let train_y = train_y.map(|x| if *x == 1 {"Survived"} else {"Not Survived"});
    // let train_y = array![train_y];
    
    let train_data = DatasetBase::new(
        train_x, 
        train_y
    );
    // println!("{}", train_data.records);
    
    // let data = DatasetView::new(train_data);
    let model = GaussianNb::params().fit(&train_data)?;
    // println!("{:?}", model);


    // 推論
    let test_x = test_x.unwrap().to_ndarray::<Float64Type>().unwrap();
    // print!("{:?}", test_x);
    // let test_x = DatasetBase::from(test_x);

    let pred = model.predict(&test_x);
    
    let pred = pred.map(|x| if *x == "Survived" {1} else {0});
    print!("{:?}", pred);


    // 混同行列の計算
    // let cm = pred.confusion_matrix(&valid)?;


    // // 混同行列と精度の出力（MCCはマシューズ相関係数）
    // println!("{:?}", cm);
    // println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
    
    Ok(())
}
