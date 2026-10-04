\set ON_ERROR_STOP on
create temp table lib(kind text, gid uuid);
\copy lib from '/tmp/lm/library_mbids.tsv'
create index on lib(gid);
-- recordings
\copy (select r.gid, r.length, r.video, r.name, ac.name from recording r join lib on lib.gid=r.gid and lib.kind='recording' join artist_credit ac on ac.id=r.artist_credit) to '/tmp/lm/rec.tsv'
-- recording -> work with performance attributes
\copy (select r.gid, w.gid, coalesce(string_agg(lat.name, ',' order by lat.name), '') from recording r join lib on lib.gid=r.gid and lib.kind='recording' join l_recording_work lrw on lrw.entity0=r.id join work w on w.id=lrw.entity1 left join link_attribute la on la.link=lrw.link left join link_attribute_type lat on lat.id=la.attribute_type group by r.gid, w.gid, lrw.id) to '/tmp/lm/rec_work.tsv'
-- recording <-> recording (both in library)
\copy (select r0.gid, r1.gid, lt.name from l_recording_recording l join link lk on lk.id=l.link join link_type lt on lt.id=lk.link_type join recording r0 on r0.id=l.entity0 join recording r1 on r1.id=l.entity1 where r0.gid in (select gid from lib where kind='recording') and r1.gid in (select gid from lib where kind='recording')) to '/tmp/lm/rec_rec.tsv'
-- artist <-> artist (both in library)
\copy (select a0.gid, a1.gid, lt.name from l_artist_artist l join link lk on lk.id=l.link join link_type lt on lt.id=lk.link_type join artist a0 on a0.id=l.entity0 join artist a1 on a1.id=l.entity1 where a0.gid in (select gid from lib where kind='artist') and a1.gid in (select gid from lib where kind='artist')) to '/tmp/lm/artist_artist.tsv'
\copy (select a.gid, coalesce(at.name,''), a.name from artist a join lib on lib.gid=a.gid and lib.kind='artist' left join artist_type at on at.id=a.type) to '/tmp/lm/artist.tsv'
-- release -> release group
\copy (select r.gid, rg.gid from release r join lib on lib.gid=r.gid and lib.kind='release' join release_group rg on rg.id=r.release_group) to '/tmp/lm/release_rg.tsv'
-- redirects: library gids that MB has merged into another entity
\copy (select 'recording', g.gid, r.gid from recording_gid_redirect g join recording r on r.id=g.new_id where g.gid in (select gid from lib) union all select 'artist', g.gid, a.gid from artist_gid_redirect g join artist a on a.id=g.new_id where g.gid in (select gid from lib) union all select 'release', g.gid, r.gid from release_gid_redirect g join release r on r.id=g.new_id where g.gid in (select gid from lib) union all select 'release-group', g.gid, r.gid from release_group_gid_redirect g join release_group r on r.id=g.new_id where g.gid in (select gid from lib)) to '/tmp/lm/redirect.tsv'
