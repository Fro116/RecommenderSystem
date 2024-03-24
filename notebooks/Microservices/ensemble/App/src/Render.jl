function dataframe_to_html_table(df::DataFrame, name)
    html = "<table id=\"$name\" class=\"display\">\n<thead>\n<tr>"
    for col_name in names(df)
        html *= "<th>$col_name</th>"
    end
    html *= "</tr>\n</thead>\n<tbody>\n"

    for row in eachrow(df)
        html *= "<tr>"
        for cell in row
            html *= "<td>$cell</td>"
        end
        html *= "</tr>\n"
    end

    html *= "<tfoot>\n<tr>"
    for col_name in names(df)
        html *= "<th>$col_name</th>"
    end
    html *= "</tr>\n</tfoot>\n"
    html *= "</tbody>\n</table>\n"
    html
end

function table_component(name)
    jquery = """
    <script>
    new DataTable('#{name}', {
        layout: {
            topStart: null,
            topEnd: null,
        	bottomStart: 'pageLength',
            bottomEnd: 'paging'
        },
        ordering: false,    
        initComplete: function () {
            this.api()
                .columns()
                .every(function () {
                    let column = this;

                    // Create input element
                    let input = document.createElement('input');
                    input.style.border = 'none';
                    input.style.outlineColor = '#546E7A';
                    input.style.outlineStyle = 'solid';
                    input.style.borderRadius = '4px';
                    input.style.color = '#e0e0e0';    
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
    """
    replace(jquery, "{name}" => name)
end

function render_html_page(username, anime_recs, manga_recs)
    imports = """
        <!doctype html>
        <html>    
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
        <link rel="stylesheet" href="https://cdn.datatables.net/2.0.1/css/dataTables.dataTables.css" />
        <script src="https://cdn.datatables.net/2.0.1/js/dataTables.js"></script>
        <style>
        tfoot input {
            width: 100%;
            padding: 3px;
            box-sizing: border-box;
        }
        html { visibility:hidden; }
        </style>
        """
    automatic_dark_mode = """
    <style>
    body {
        background-color: #22272e; 
        color: #e0e0e0; 
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0; /* Slightly brighter for emphasis */
    }


    a {
        color: #90caf9; /* Light blue for links */
        text-decoration: none; /* Remove underline by default */
    }

    a:hover {
        color: #c5e1f2; /* Slightly brighter version of the light blue */
        text-decoration: underline; /* Add underline on hover */
    }

    /* Form elements */
    input, textarea, select {
        background-color: #333;
        color: #e0e0e0;
        border-color: #555;
    }

    /* Images - Consider more advanced techniques if needed */
    img {
        filter: brightness(80%) contrast(110%); 
    }

    /* Styling for specific components or elements */
    /* Example: */
    .card {
        background-color: #333;
        border-color: #444;
    }

    </style>
    <script>
    let prefers = 'dark';
    let html = document.querySelector('html');
    html.classList.add(prefers);
    html.setAttribute('data-bs-theme', prefers);
    \$(document).ready(function() {
        document.getElementsByTagName("html")[0].style.visibility = "visible";
    });
    </script>
    """
    title = "<title>$username's Recs</title>"
    io = IOBuffer()    
    write(io, imports)
    write(io, automatic_dark_mode)        
    write(io, title)
    write(io, dataframe_to_html_table(anime_recs, "anime"))
    write(io, dataframe_to_html_table(manga_recs, "manga"))
    write(io, table_component("anime"))
    write(io, table_component("manga"))
    write(io, "</html>")
    String(take!(io))
end