#!/bin/bash

kill_bg_processes() {
  trap '' INT TERM
  kill -INT 0
  wait
}

trap kill_bg_processes INT
logs="../../logs/collect"
mkdir -p $logs && rm $logs/*.log
(uvicorn layer1:app --host 0.0.0.0 --port 4101 --log-level warning |& julia -t 1 logrotate.jl $logs/layer1.1.log) &
(uvicorn layer1:app --host 0.0.0.0 --port 4102 --log-level warning |& julia -t 1 logrotate.jl $logs/layer1.2.log) &
(uvicorn layer1:app --host 0.0.0.0 --port 4103 --log-level warning |& julia -t 1 logrotate.jl $logs/layer1.3.log) &
(uvicorn layer1:app --host 0.0.0.0 --port 4104 --log-level warning |& julia -t 1 logrotate.jl $logs/layer1.4.log) &
(julia layer2.jl 4002 1 "http://localhost:4101/proxy,http://localhost:4102/proxy,http://localhost:4103/proxy,http://localhost:4104/proxy" true 10 true |& julia -t 1 logrotate.jl $logs/layer2.log) &
(julia layer3.jl 4003 "http://localhost:4002" 1000 5 |& julia -t 1 logrotate.jl $logs/layer3.log) &

(julia collect_single.jl mal_userids userid "http://localhost:4003/mal_username" 1 |& julia -t 1 logrotate.jl $logs/mal_userids.log) &
(julia collect_junction.jl mal_users user mal_user_items items mal_userids userid username db_junction_last_changed_at "http://localhost:4003/mal_user" 150 |& julia -t 1 logrotate.jl $logs/mal_users.log) &
(julia collect_junction.jl mal_media details mal_media_relations relations mal_user_items nothing medium,itemid db_primary_last_changed_at "http://localhost:4003/mal_media" 1 |& julia -t 1 logrotate.jl $logs/mal_media.log) &

(julia collect_junction.jl anilist_users user anilist_user_items items nothing nothing userid db_junction_last_changed_at "http://localhost:4003/anilist_user" 30 |& julia -t 1 logrotate.jl $logs/anilist_users.log) &
(julia collect_junction.jl anilist_media details anilist_media_relations relations anilist_user_items nothing medium,itemid db_primary_last_changed_at "http://localhost:4003/anilist_media" 1 |& julia -t 1 logrotate.jl $logs/anilist_media.log) &

(julia collect_junction.jl kitsu_users user kitsu_user_items items nothing nothing userid db_junction_last_changed_at "http://localhost:4003/kitsu_user" 10 |& julia -t 1 logrotate.jl $logs/kitsu_users.log) &
(julia collect_junction.jl kitsu_media details kitsu_media_relations relations kitsu_user_items nothing medium,itemid db_primary_last_changed_at "http://localhost:4003/kitsu_media" 1 |& julia -t 1 logrotate.jl $logs/kitsu_media.log) &

(julia collect_single.jl animeplanet_userids userid "http://localhost:4003/animeplanet_username" 1 |& julia -t 1 logrotate.jl $logs/animeplanet_userids.log) &
(julia collect_junction.jl animeplanet_users user animeplanet_user_items items animeplanet_userids userid username db_junction_last_changed_at "http://localhost:4003/animeplanet_user" 80 |& julia -t 1 logrotate.jl $logs/animeplanet_users.log) &
(julia collect_junction.jl animeplanet_media details animeplanet_media_relations relations animeplanet_user_items nothing medium,itemid db_primary_last_changed_at "http://localhost:4003/animeplanet_media" 1 |& julia -t 1 logrotate.jl $logs/animeplanet_media.log) &

(julia -t 1 collect_external.jl |& julia -t 1 logrotate.jl $logs/external.log) &

(julia backup.jl |& julia -t 1 logrotate.jl $logs/backup.log) &

tail -F $logs/layer1.1.log $logs/layer1.2.log $logs/layer1.3.log $logs/layer1.4.log $logs/layer2.log $logs/layer3.log $logs/mal_userids.log $logs/animeplanet_userids.log $logs/mal_users.log $logs/animeplanet_users.log $logs/anilist_users.log $logs/kitsu_users.log $logs/mal_media.log $logs/anilist_media.log $logs/kitsu_media.log $logs/animeplanet_media.log $logs/backup.log $logs/external.log
