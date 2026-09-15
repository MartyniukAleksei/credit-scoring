from pydantic import BaseModel, Field

class BorrowerFeatures(BaseModel):
    revolving_utilization_of_unsecured_lines: float = Field(
        ..., alias="RevolvingUtilizationOfUnsecuredLines", ge=0
    )
    age: int = Field(..., ge=0)
    number_of_time_30_59_days_past_due_not_worse: int = Field(
        ..., alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0
    )
    debt_ratio: float = Field(..., alias="DebtRatio", ge=0)
    monthly_income: float | None = Field(
        default=None, alias="MonthlyIncome", ge=0
    )
    number_of_open_credit_lines_and_loans: int = Field(
        ..., alias="NumberOfOpenCreditLinesAndLoans", ge=0
    )
    number_of_times_90_days_late: int = Field(
        ..., alias="NumberOfTimes90DaysLate", ge=0
    )
    number_real_estate_loans_or_lines: int = Field(
        ..., alias="NumberRealEstateLoansOrLines", ge=0
    )
    number_of_time_60_89_days_past_due_not_worse: int = Field(
        ..., alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0
    )
    number_of_dependents: float | None = Field(
        default=None, alias="NumberOfDependents", ge=0
    )
    
