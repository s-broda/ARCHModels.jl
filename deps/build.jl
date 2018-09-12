datadir = joinpath(@__DIR__, "..", "src", "data")
mkdir(datadir)
download("http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF20-1.txt", joinpath(datadir, "bollerslev_ghysels.txt"))
