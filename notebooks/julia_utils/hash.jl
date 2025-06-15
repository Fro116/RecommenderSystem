import SHA

shahash(obj) = parse(BigInt, bytes2hex(SHA.sha256(string(obj))), base=16)
stringhash(obj::Vector{UInt8}) = bytes2hex(SHA.sha256(obj))
