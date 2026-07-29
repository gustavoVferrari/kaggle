import pandas as pd
import great_expectations as gx


class DataValidator:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.gx_df = gx.from_pandas(dataframe)

    def validate(self):

        # Colunas obrigatórias
        self.gx_df.expect_table_columns_to_match_ordered_list([
            "LotArea",
            "OverallQual",
            "YearBuilt",
            "GrLivArea",
            "SalePrice"
        ])

        # Não pode haver valores nulos
        self.gx_df.expect_column_values_to_not_be_null("LotArea")
        self.gx_df.expect_column_values_to_not_be_null("OverallQual")
        self.gx_df.expect_column_values_to_not_be_null("GrLivArea")

        # Valores mínimos
        self.gx_df.expect_column_values_to_be_between(
            "LotArea",
            min_value=100
        )

        self.gx_df.expect_column_values_to_be_between(
            "OverallQual",
            min_value=1,
            max_value=10
        )

        self.gx_df.expect_column_values_to_be_between(
            "SalePrice",
            min_value=10000
        )

        # Tipo de dados
        self.gx_df.expect_column_values_to_be_of_type(
            "OverallQual",
            "int64"
        )

        # Percentual máximo de nulos
        self.gx_df.expect_column_proportion_of_non_null_values_to_be_between(
            "LotArea",
            min_value=0.99
        )

        return self.gx_df.validate()