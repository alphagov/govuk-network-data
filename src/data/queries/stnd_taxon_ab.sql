SELECT
  COUNT(*) AS Occurrences,
  REPLACE(ab_variant,"RelatedLinksAATest:","") as ABVariant,
  STRING_AGG(DeviceCategory, ",") AS DeviceCategories,
  Sequence
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
          NULL), ">>") OVER (PARTITION BY fullVisitorId, visitId ORDER BY hitNumber ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING ) AS PageSequence,
      DeviceCategory,
      ab_variant
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
        (
        SELECT
          IFNULL(value,"NULL")
        FROM
          sessions.customDimensions
        WHERE
          index=65) AS ab_variant,
        device.deviceCategory AS DeviceCategory
      FROM
        `govuk-bigquery-analytics.87773428.ga_sessions_TIME_STAMP` AS sessions
      CROSS JOIN
        UNNEST(sessions.hits) AS hits ) )
  WHERE
    DeviceCategory != "tablet" and ab_variant != "NULL"
  GROUP BY
    sessionId,
    Sequence,
    PageSequence,
    DeviceCategory,
    ab_variant)
GROUP BY
  Sequence,
  ABVariant