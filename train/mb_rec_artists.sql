\set ON_ERROR_STOP on
create temp table lib(kind text, gid uuid);
\copy lib from '/tmp/lm/library_mbids.tsv'
\copy (select r.gid, string_agg(distinct a.gid::text, ',') from recording r join lib on lib.gid = r.gid and lib.kind = 'recording' join artist_credit_name acn on acn.artist_credit = r.artist_credit join artist a on a.id = acn.artist group by r.gid) to '/tmp/lm/rec_artists.tsv'
