\set ON_ERROR_STOP on
create temp table lib(kind text, gid uuid);
\copy lib from '/tmp/lm/library_mbids.tsv'
create temp table librec as select r.id, r.gid from recording r join lib on lib.gid = r.gid and lib.kind = 'recording';
create index on librec(id);
-- each library recording's own first release year
\copy (select lr.gid, f.year from librec lr join recording_first_release_date f on f.recording = lr.id where f.year is not null) to '/tmp/lm/rec_year.tsv'
-- works of library recordings: first year any recording of the work was released
create temp table libwork as select distinct lrw.entity1 as work from l_recording_work lrw join librec lr on lr.id = lrw.entity0;
\copy (select w.gid, min(f.year) from libwork lw join work w on w.id = lw.work join l_recording_work lrw on lrw.entity1 = lw.work join recording_first_release_date f on f.recording = lrw.entity0 where f.year is not null group by w.gid) to '/tmp/lm/work_year.tsv'
