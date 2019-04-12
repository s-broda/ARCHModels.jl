using Retry
datadir = joinpath(@__DIR__, "..", "src", "data")
isdir(datadir) || mkdir(datadir)
@info "Downloading Bollerslev and Ghysels data..."
isfile(joinpath(datadir, "bollerslev_ghysels.txt")) || download("http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF20-1.txt", joinpath(datadir, "bollerslev_ghysels.txt"))

@info "Downloading stock data..."
#"DOW" is excluded because it's listed too late
tickers = ["AAPL", "IBM", "XOM", "KO", "MSFT", "INTC", "MRK", "PG", "VZ", "WBA", "V", "JNJ", "PFE", "CSCO", "TRV", "WMT", "MMM", "UTX", "UNH", "NKE", "HD", "BA", "AXP", "MCD", "CAT", "GS", "JPM", "CVX", "DIS"]
for (j, ticker) in enumerate(tickers)
        @repeat 4 try
            isfile(joinpath(datadir, "$ticker.csv")) || download("http://quotes.wsj.com/$ticker/historical-prices/download?num_rows=100000000&range_days=100000000&startDate=03/19/2008&endDate=04/11/2019", joinpath(datadir, "$ticker.csv"))
            data = parse.(Float64, readdlm(joinpath(dirname(pathof(ARCHModels)), "data", "$ticker.csv"), ',', String, skipstart=1)[:, 5])
            if ticker == "CSCO" #wsj misses the quote for June 14th, 2010
                insert!(data, 2223, 22.76)
            end
            length(data) == 2786 || error("Download failed for $ticker.")
        catch e
            @delay_retry if 1==1 end
        end
end
