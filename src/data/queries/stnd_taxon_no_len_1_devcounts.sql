SELECT
  COUNT(*) AS Occurrences,
  SUM(IF(DeviceCategory='mobile',1,0)) AS MobileCount,
  SUM(IF(DeviceCategory='desktop',1,0)) AS DesktopCount,
  PageSeq_Length,
  Actions_Length,
  Sequence,
  PageSequence
FROM (
  SELECT
    *
  FROM (
    SELECT
      CONCAT(fullVisitorId,"-",CAST(visitId AS STRING),"-",CAST(visitNumber AS STRING)) AS sessionId,
      STRING_AGG(CONCAT(pagePath,"<<",CONCAT(htype,"<:<",IFNULL(eventCategory,
              "NULL"),"<:<",IFNULL(eventAction,
              "NULL")),"<<",taxon), ">>") OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS Sequence,
      STRING_AGG(IF(htype = 'PAGE',
          pagePath,
          NULL),">>") OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS PageSequence,
      DeviceCategory,
      COUNT(*) OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS Actions_Length,
      SUM(IF(htype='PAGE',
          1,
          0)) OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS PageSeq_Length
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
        (
        SELECT
          value
        FROM
          hits.customDimensions
        WHERE
          index=59) AS taxon,
        device.deviceCategory AS DeviceCategory
      FROM
        `govuk-bigquery-analytics.87773428.ga_sessions_TIME_STAMP` AS sessions
      CROSS JOIN
        UNNEST(sessions.hits) AS hits )
)
WHERE
      PageSeq_Length >1
  GROUP BY
    sessionId,
    Sequence,
    PageSequence,
    DeviceCategory,
    Actions_Length,
    PageSeq_Length)
GROUP BY
  Sequence,
  PageSequence,
  PageSeq_Length,
  Actions_Length