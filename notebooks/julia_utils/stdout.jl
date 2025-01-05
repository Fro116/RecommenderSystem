import Dates

STDOUT_LOCK::ReentrantLock = ReentrantLock()

function logtag(tag::AbstractString, x::AbstractString)
    lock(STDOUT_LOCK) do
        println("$(Dates.now()) [$tag] $x")
    end
end

logerror(x::AbstractString) = logtag("ERROR", x)
