import CSV
import DataFrames
import Dates
import HTTP
import JSON
import Markdown: @md_str
import NBInclude: @nbinclude
import Oxygen
@nbinclude("../TrainingAlphas/Alpha.ipynb")

# Get media

function to_hyperlink(title, url)
    "<a href=\"$url\" target=\"_blank\">$title</a>"
end

function get_hyperlink(title, links, source)
    if source == "mal"
        search = "myanimelist"
    elseif source == "anilist"
        search = "anilist"
    elseif source == "kitsu"
        search = "kitsu"
    elseif source == "animeplanet"
        search = "anime-planet"
    else
        @assert false
    end

    # try to return the preferred source
    parsed_links = JSON.parse(replace(links, "'" => "\""))
    for s in [search, "myanimelist"]
        for link in parsed_links
            if occursin(s, link)
                return to_hyperlink(title, link)
            end
        end
    end
    title
end

function parse_genres(x)
    genres = JSON.parse(replace(x, "'" => "\"", "_" => " "))
    join(genres, ", ")
end

function get_media(medium::String, source::String)
    df = DataFrame(
        CSV.File(
            get_data_path("processed_data/$medium.csv"),
            ntasks = 1;
            stringtype = String,
        ),
    )
    df.title = get_hyperlink.(df.title, df.links, source)
    df.genres = parse_genres.(df.genres)
    # validate fields
    valid_statuses = Dict(
        "anime" => ["Currently Airing", "Finished Airing", "Not yet aired"],
        "manga" => [
            "Not yet published",
            "Publishing",
            "Discontinued",
            "Finished",
            "On Hiatus",
        ],
    )
    @assert issubset(Set(df.status), Set(valid_statuses[medium]))
    df
end

function prune_media_df(df, medium)
    if medium == "anime"
        series_length = ["num_episodes"]
    elseif medium == "manga"
        series_length = ["num_volumes", "num_chapters"]
    end
    keepcols = vcat(
        ["mediaid", "uid", "title", "type"],
        series_length,
        ["status", "start_date", "end_date", "genres", "tags", "summary"],
    )
    df[:, keepcols]
end

@memoize function get_media_df(medium, source)
    media = get_media(medium, source)
    media_to_uid = DataFrame(CSV.File(get_data_path("processed_data/$(medium)_to_uid.csv")))
    df = DataFrames.innerjoin(media_to_uid, media, on = "mediaid" => "$(medium)_id")
    prune_media_df(df, medium)
end

# Get rankings

function get_rating_df(medium, username, source)
    get_alpha(x) =
        get_raw_split("rec_inference", medium, [:itemid], x, username, source).alpha
    rating_df = DataFrame(
        "uid" => 0:num_items(medium)-1,
        "rating" => get_alpha("$medium/Linear/rating"),
        "watch" => get_alpha("$medium/Linear/watch"),
        "plantowatch" => get_alpha("$medium/Linear/plantowatch"),
        "drop" => get_alpha("$medium/Linear/drop"),
        "num_dependencies" => get_alpha("$medium/Nondirectional/Dependencies"),
        "is_sequel" => get_alpha("$medium/Nondirectional/SequelSeries"),
        "is_direct_sequel" => get_alpha("$medium/Nondirectional/DirectSequelSeries"),
        "is_related" => get_alpha("$medium/Nondirectional/RelatedSeries"),
        "is_recap" => get_alpha("$medium/Nondirectional/RecapSeries"),
        "is_cross_recap" => get_alpha("$medium/Nondirectional/CrossRecapSeries"),
        "is_cross_related" => get_alpha("$medium/Nondirectional/CrossRelatedSeries"),
    )
    rating_df[:, "score"] .= (
        rating_df.rating +
        (log.(rating_df.watch) ./ log(10)) +
        0.1 * (log.(rating_df.plantowatch) ./ log(10)) +
        (-max.(rating_df.drop, 0.01) * 10)
    )

    rating_df[:, "seen"] .= false
    seen_df = get_raw_split("rec_training", medium, [:itemid], nothing, username, source)
    rating_df.seen[seen_df.itemid.+1] .= true
    rating_df[:, "ptw"] .= false
    ptw_df = get_split(
        "rec_training",
        "plantowatch",
        medium,
        [:itemid],
        nothing,
        username,
        source,
    )
    rating_df.ptw[ptw_df.itemid.+1] .= true
    rating_df.seen[ptw_df.itemid.+1] .= false
    rating_df
end

function get_ranking_df(medium, username, source)
    rating_df = get_rating_df(medium, username, source)
    media_df = get_media_df(medium, source)
    DataFrames.innerjoin(media_df, rating_df, on = "uid")
end

# Display options

# filters generics
display_filter(f) = (df; fn = identity) -> filter(fn ∘ f, df)
inv(f) = (x...) -> f(x...; fn = !);

# filter by related series
seen = display_filter(x -> x.seen) # item has been seen
related = display_filter(x -> x.related != 0) # item is related to a seen item
cross_related = display_filter(x -> x.is_cross_related != 0) # item is related to a seen item in a different media
recap = display_filter(x -> x.is_recap != 0 && x.is_direct_sequel == 0) # item is a recap of a seen item
cross_recap = display_filter(x -> x.is_cross_recap != 0 && x.is_direct_sequel == 0) # item is a recap of a seen item in a different media
dependent = display_filter(x -> x.num_dependencies > 0 && x.is_direct_sequel == 0); # item is a sequel that we havent seen the prequel for

# filter by date
function parse_date(x)
    if x in ["Not available"]
        return nothing
    end
    fields = split(x, " ")
    if length(fields) == 3
        date_format = "u d, Y"
    elseif length(fields) == 2
        date_format = "u Y"
    elseif length(fields) == 1
        date_format = "Y"
    else
        @assert false x
    end
    parsed_date = Dates.DateTime(x, date_format)
    Int(Dates.datetime2unix(parsed_date))
end

function released_after(x, timestamp)
    release_date = parse_date(x.start_date)
    if isnothing(release_date)
        return false
    else
        return timestamp < release_date
    end
end

function released_before(x, timestamp)
    release_date = parse_date(x.start_date)
    if isnothing(release_date)
        return false
    else
        return release_date < timestamp
    end
end

after(year, month = 1, date = 1) = display_filter(
    x -> released_after(x, Dates.datetime2unix(Dates.DateTime(year, month, date))),
)
before(year, month = 1, date = 1) = display_filter(
    x -> released_before(x, Dates.datetime2unix(Dates.DateTime(year, month, date))),
)
status(s) = display_filter(x -> x.status == s)
function released(medium)
    if medium == "anime"
        return inv(status("Not yet aired"))
    elseif medium == "manga"
        return inv(status("Not yet published"))
    else
        @assert false
    end
end

# filter by content
max_episodes(n) = display_filter(x -> x.num_episodes <= n)
search(key::String, col) = display_filter(x -> occursin(lowercase(key), lowercase(x[col])))
search(key, col) = display_filter(x -> x[col] == key)
search(key::Vector, col) = display_filter(x -> x[col] in key)
search(key::String) = search(key, :title);

# filter by score
head(n) = x -> first(x, n);
top(n, field) = x -> first(sort(x, field, rev = true), n)
top(n) = top(n, :score);

function format(df::DataFrame, medium::String, debug::Bool)
    df[!, ""] = 1:length(df.title)
    if debug
        df = DataFrames.select(df, DataFrames.Not([:summary, :tags]))
    else
        if medium == "anime"
            cols = ["", "title", "type", "num_episodes", "start_date", "genres"]
        elseif medium == "manga"
            cols = ["", "title", "type", "status", "start_date", "genres"]
        else
            @assert false
        end
        df = DataFrames.select(df, Symbol.(cols))
    end
    headers = titlecase.(replace.(names(df), "_" => " "))
    headers = replace(
        headers,
        "Title" => titlecase(medium),
        "Num Episodes" => "Episodes",
        "Num Volumes" => "Volumes",
        "Num Chapters" => "Chapters",
    )
    DataFrames.rename!(df, headers)
    df
end

format(medium::String, debug::Bool) = x -> format(x, medium, debug);

# Render in jupyterlab

ENV["DATAFRAMES_COLUMNS"] = 300
ENV["DATAFRAMES_ROWS"] = 300

function html(df)
    Base.show(
        stdout,
        MIME("text/html"),
        df;
        allow_html_in_cells = true,
        header = names(df),
        show_row_number = false,
        row_number_column_title = "Rank",
        top_left_str = "",
    )
end

# Render HTML

function dataframe_to_html_table(df::DataFrame, name)
    # Start the table and add a header row
    html = "<table id=\"$name\" class=\"display\">\n<thead>\n<tr>"
    for col_name in names(df)
        html *= "<th>$col_name</th>"
    end
    html *= "</tr>\n</thead>\n<tbody>\n"

    # Add table rows for each entry in the DataFrame
    for row in eachrow(df)
        html *= "<tr>"
        for cell in row
            html *= "<td>$cell</td>"
        end
        html *= "</tr>\n"
    end

    # add footer
    html *= "<tfoot>\n<tr>"
    for col_name in names(df)
        html *= "<th>$col_name</th>"
    end
    html *= "</tr>\n</tfoot>\n"

    # Close the table
    html *= "</tbody>\n</table>"

    html
end

function table_component(name)
    jquery = raw"<script>
new DataTable('#{name}', {
    initComplete: function () {
        this.api()
            .columns()
            .every(function () {
                let column = this;
                let title = column.footer().textContent;
 
                // Create input element
                let input = document.createElement('input');
                input.placeholder = title;
                column.footer().replaceChildren(input);
 
                // Event listener for user input
                input.addEventListener('keyup', () => {
                    if (column.search() !== this.value) {
                        column.search(input.value).draw();
                    }
                });
            });
    }
});
</script>
"
    replace(jquery, "{name}" => name)
end

function render_html_page(fn, username, anime_recs, manga_recs)
    imports = """
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/2.0.1/css/dataTables.dataTables.css" />
<script src="https://cdn.datatables.net/2.0.1/js/dataTables.js"></script>

<style>
tfoot input {
        width: 100%;
        padding: 3px;
        box-sizing: border-box;
    }
</style>

    """
    title = "<title>$username's Recs</title>"
    open(fn, "w") do io
        write(io, imports)
        write(io, title)
        write(io, dataframe_to_html_table(anime_recs, "anime"))
        write(io, table_component("anime"))
        write(io, dataframe_to_html_table(manga_recs, "manga"))
        write(io, table_component("manga"))
    end
end

# Recommendations

function recommend(
    medium,
    username,
    source;
    M = 1000,
    N = 100,
    debug = false,
    extra_filters = identity,
)
    get_ranking_df(medium, username, source) |>
    extra_filters |>
    released(medium) |>
    inv(seen) |>
    inv(recap) |>
    inv(cross_recap) |>
    inv(dependent) |>
    top(1000, :watch) |>
    top(100, :score) |>
    format(medium, debug)
end

function runscript(username, source)
    fn = get_data_path("recommendations/$source/$username/Recommendations.html")
    anime_recs = recommend("anime", username, source)
    manga_recs = recommend("manga", username, source)
    render_html_page(fn, username, anime_recs, manga_recs)
end

Oxygen.@get "/query" function(req::HTTP.Request)
    params = Oxygen.queryparams(req)
    runscript(params["username"], params["source"])
end

Oxygen.serveparallel(; port=parse(Int, ARGS[1]))