SELECT
  COUNT(*) AS Occurrences,
  STRING_AGG(DeviceCategory,",") AS DeviceCategories,
  PageSeq_Length,
  Actions_Length,
  STRING_AGG(Date,",") AS Dates,
  Sequence
FROM (
  SELECT
    *
  FROM (
    SELECT
      CONCAT(fullVisitorId,"-",CAST(visitId AS STRING),"-",CAST(visitNumber AS STRING)) AS sessionId,
      STRING_AGG(CONCAT(pagePath,"<<",CONCAT(IFNULL(eventCategory, "NULL"),"<:<",IFNULL(eventAction, "NULL"))),
        ">>") OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS Sequence,
      DeviceCategory,
      Date,
      COUNT(*) OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS Actions_Length,
      SUM(IF(htype='PAGE', 1, 0)) OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS PageSeq_Length,
      SUM(IF(eventAction='ffYesClick', 1, 0)) OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS EventYes
    FROM (
      SELECT
        fullVisitorId,
        visitId,
        visitNumber,
        visitStartTime,
        hits.page.pagePath AS pagePath,
        hits.hitNumber AS hitNumber,
        hits.type AS htype,
        hits.eventInfo.eventAction AS eventAction,
        hits.eventInfo.eventCategory AS eventCategory,
        date AS Date,
        device.deviceCategory AS DeviceCategory
      FROM
        `govuk-bigquery-analytics.87773428.ga_sessions_TIME_STAMP` AS sessions
      CROSS JOIN
        UNNEST(sessions.hits) AS hits ))
  GROUP BY
    sessionId,
    Sequence,
    DeviceCategory,
    Date,
    EventYes,
    Actions_Length,
    PageSeq_Length)
GROUP BY
  Sequence,
  PageSeq_Length,
  Actions_Length