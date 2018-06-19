"""
New items are pushed to the back of the list, overwriting values in a circular fashion.
"""
mutable struct CircularBuffer{T} <: AbstractVector{T}
    capacity::Int
    first::Int
    buffer::Vector{T}
    CircularBuffer{T}(capacity::Int) where {T} = new{T}(capacity, 1, zeros(T, capacity))
end



@inline function _buffer_index_checked(cb::CircularBuffer, i::Int)
    n = cb.capacity
    idx = cb.first + i - 1
    if idx > n
        idx - n
    else
        idx
    end
end

@inline function Base.getindex(cb::CircularBuffer, i::Int)
    cb.buffer[_buffer_index_checked(cb, i)]
end

@inline function Base.setindex!(cb::CircularBuffer, data, i::Int)
    cb.buffer[_buffer_index_checked(cb, i)] = data
    cb
end


@inline function Base.push!(cb::CircularBuffer, data)
    cb.first = (cb.first == cb.capacity ? 1 : cb.first + 1)
    cb.buffer[_buffer_index_checked(cb, cb.capacity)] = data
    cb
end

@inline Base.length(cb::CircularBuffer) = cb.capacity
@inline Base.size(cb::CircularBuffer) = (length(cb),)
@inline Base.convert(::Type{Array}, cb::CircularBuffer{T}) where {T} = T[x for x in cb]
@inline Base.isempty(cb::CircularBuffer) = false

@inline capacity(cb::CircularBuffer) = cb.capacity
@inline isfull(cb::CircularBuffer) = length(cb) == cb.capacity
