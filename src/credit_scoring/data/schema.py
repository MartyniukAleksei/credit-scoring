import pandera.pandas as pa
from pandera import Column,  Check, DataFrameSchema

DEBT_RATIO_CAP=2.5
REVOLVING_UTIL_CAP=10

base_schema = DataFrameSchema({
    "age": Column(
        int, 
        checks=[
            Check.ge(0),
            Check.ne(0, raise_warning=True)
        ],
        nullable=False,
    ),
    "MonthlyIncome": Column(
        float, 
        checks=[
            Check.ge(0),
            Check(
                lambda s: s.isna().mean() <= 0.25,
                element_wise=False,
                ignore_na=False,
                raise_warning=True,
                error="Percent of NaN in MonthlyIncome more than 20%"
            )
        ],
        nullable=True
    ),
    "RevolvingUtilizationOfUnsecuredLines": Column(
        float, 
        checks=[
            Check.ge(0),
            Check(
                lambda s: (s > REVOLVING_UTIL_CAP).mean() <= 0.05,
                element_wise=False,
                ignore_na=False,
                raise_warning=True,
                error="Percent RevolvingUtilization more than historic norm value"
            )
        ],
        nullable=False
    ),
    "NumberOfTime30-59DaysPastDueNotWorse": Column(
        int,
        checks=[
            Check.ge(0),
            Check.notin([96, 98], raise_warning=True)
        ],
        nullable=False
    ),
    "DebtRatio": Column(
        float,
        checks=[
            Check.ge(0),
            Check(
                lambda s: (s > DEBT_RATIO_CAP).mean() <= 0.05, 
                element_wise=False,
                ignore_na=False,
                raise_warning=True,
                error="Percent DebtRatio more than historical norm value"
            )
        ],
        nullable=False
    ),
    "NumberOfOpenCreditLinesAndLoans": Column(
        int,
        checks=[
            Check.ge(0)
        ],
        nullable=False
    ),
    "NumberOfTimes90DaysLate": Column(
        int,
        checks=[
            Check.ge(0),
            Check.notin([98, 96], raise_warning=True)
        ],
        nullable=False
    ),
    "NumberRealEstateLoansOrLines": Column(
        int,
        checks=[
            Check.ge(0)
        ],
        nullable=False
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": Column(
        int,
        checks=[
            Check.ge(0),
            Check.notin([96, 98], raise_warning=True)
        ],
        nullable=False
    ),
    "NumberOfDependents": Column(
        float,
        checks=[
            Check.ge(0),
            Check(
                lambda s: s.isna().mean() <= 0.10,
                element_wise=False,
                ignore_na=False,
                raise_warning=True,
                error="Percent of NaN in NumberOfDependents more than 10%"
            )
        ],
        nullable=True
    )
})

train_schema = base_schema.add_columns({
    "SeriousDlqin2yrs": Column(int, Check.isin([0,1]), nullable=False)
})