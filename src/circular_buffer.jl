Base.@propagate_inbounds function _buffer_index_checked(cb::CircularBuffer, i::Int)
    @boundscheck if i < 1 || i > cb.length
        throw(BoundsError(cb, i))
    end
    _buffer_index(cb, i)
end

@inline function _buffer_index(cb::CircularBuffer, i::Int)
    n = cb.capacity
    idx = cb.first + i - 1
    if idx > n
        idx - n
    else
        idx
    end
end

@inline Base.@propagate_inbounds function Base.getindex(cb::CircularBuffer, i::Int)
    cb.buffer[_buffer_index_checked(cb, i)]
end

@inline Base.@propagate_inbounds function Base.setindex!(cb::CircularBuffer, data, i::Int)
    cb.buffer[_buffer_index_checked(cb, i)] = data
    cb
end

@inline function Base.push!(cb::CircularBuffer, data)
    # if full, increment and overwrite, otherwise push
    if cb.length == cb.capacity
        cb.first = (cb.first == cb.capacity ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    cb.buffer[_buffer_index(cb, cb.length)] = data
    cb
end
