SELECT *
FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_],
    TIME_STAMP))
    WHERE PageSeq_Length > 1
