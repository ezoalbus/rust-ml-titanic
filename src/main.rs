use linfa::prelude::*;
use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::GaussianNb;
use linfa_bayes::Result as NBResult;

use polars::prelude::*;
use polars::prelude::Result as PolarResult;


fn read_csv2df(path: &str) -> PolarResult<DataFrame> {
    // csvを読み込んで、DataFrameでreturn
    CsvReader::from_path(path)?
            .has_header(true)
            .finish()
}

fn split_x_y(df: &DataFrame) -> (PolarResult<DataFrame>, PolarResult<DataFrame>) {
    // 特徴量とターゲットに分割
    let target = df.select(vec!["Survived"]);
    // 運賃の情報だけを特徴量として使う
    let features = df.select(vec!["Fare"]);
    (features, target)
}

fn main() -> NBResult<()> {
    //  trainの読み込み
    let train_path = "./data/train.csv";
    let train_df = read_csv2df(&train_path).unwrap();

    // 特徴量とターゲットに分割
    let (train_x, train_y) = split_x_y(&train_df);

    // 前処理
    let train_x = train_x.unwrap().to_ndarray::<Float64Type>().unwrap();
    let train_y = train_y.unwrap().to_ndarray::<Int64Type>().unwrap();
    let train_y = train_y.map(|x| if *x == 1 {"Survived"} else {"Not Survived"});

    // DatasetBaseとしてまとめ、trainとvalidに分割する
    let (train, valid) = DatasetBase::new(
        train_x, 
        train_y
    ).split_with_ratio(0.8);
    
    // 訓練
    let model = GaussianNb::params().fit(&train)?;

    // 推論
    let pred = model.predict(&valid);
    
    // 混同行列の計算
    let cm = pred.confusion_matrix(&valid)?;

    // 混同行列と精度の出力（MCCはマシューズ相関係数）
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
    
    Ok(())
}
