SELECT
  COUNT(*) AS Occurrences,
  PageSeq_Length,
  Actions_Length,
  GROUP_CONCAT(TrafficSource,",") AS TrafficSources,
  GROUP_CONCAT(TrafficMedium,",") AS TrafficMediums,
  Sequence
FROM (
  SELECT
    *
  FROM (
    SELECT
      CONCAT(fullVisitorId,"-",STRING(visitId),"-",STRING(visitNumber),"-",STRING(TIMESTAMP(INTEGER(visitStartTime*1000000)))) AS sessionId,
      GROUP_CONCAT(CONCAT(pagePath,"::",CONCAT(IFNULL(hits.eventInfo.eventCategory,"NULL"),"//",IFNULL(hits.eventInfo.eventAction,"NULL"))),">>") OVER (PARTITION BY fullVisitorId, visitId ORDER BY hits.hitNumber rows BETWEEN unbounded preceding AND unbounded following ) AS Sequence,
      TrafficSource,
      TrafficMedium,
      Date,
      COUNT(*) OVER (PARTITION BY fullVisitorId, visitId ORDER BY hits.hitNumber rows BETWEEN unbounded preceding AND unbounded following ) AS Actions_Length,
      SUM(IF(hits.type='PAGE',1,0)) OVER (PARTITION BY fullVisitorId, visitId ORDER BY hits.hitNumber rows BETWEEN unbounded preceding AND unbounded following ) AS PageSeq_Length
    FROM (
      SELECT
        fullVisitorId,
        visitId,
        visitNumber,
        visitStartTime,
        hits.page.pagePath AS pagePath,
        hits.hitNumber AS hitNumber,
        trafficSource.source AS TrafficSource,
        trafficSource.medium AS TrafficMedium,
        hits.eventInfo.eventAction,
        date AS Date,
        hits.type,
        hits.eventInfo.eventCategory,
      FROM
        TABLE_DATE_RANGE([govuk-bigquery-analytics:87773428.ga_sessions_],
          TIME_STAMP))
    WHERE
      PageSeq_Length > 1
  GROUP BY
    sessionId,
    Sequence,
    TrafficSource,
    TrafficMedium,
    Date,
    Actions_Length,
    PageSeq_Length)
GROUP BY
  Sequence,
  PageSeq_Length,
  Actions_Length