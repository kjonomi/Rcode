##############################################################
# Regenerates beijing_station_graph.png from the current
# pipeline (Real.R), replacing the version committed
# during the Real.R -> Real.R rewrite and never rebuilt
# since. Uses the default SYMM=max convention.
#
#   Rscript make_beijing_graph_figure.R
##############################################################

suppressMessages({library(ggplot2); library(dplyr)})

GC  <- if (nzchar(Sys.getenv("GC_DIR"))) Sys.getenv("GC_DIR") else getwd()
OUT <- GC   # figures land beside the code
Sys.setenv(GC_DIR = GC, STATION_ORDER = "spatial", SYMM = "max", SEED = "1")

## Source Real.R only up to where A reaches its final symmetrized form
## (line with `cat(sprintf("\n>> symmetrization: %s\n", SYMM))`), well before
## any model is built or trained.
src   <- readLines(file.path(GC, "Real.R"), warn = FALSE)
stop_at <- grep('cat\\(sprintf\\("\\\\n>> symmetrization', src, fixed = FALSE)[1]
stopifnot(!is.na(stop_at))
eval(parse(text = paste(src[1:(stop_at - 1)], collapse = "\n")), envir = globalenv())

stopifnot(exists("A"), exists("station_coordinates"), nrow(station_coordinates) == 12)

## Edge list: undirected, one row per pair with a positive weight.
N_STATIONS <- nrow(station_coordinates)
edge_df <- data.frame()
for (i in seq_len(N_STATIONS)) {
  for (j in seq_len(N_STATIONS)) {
    if (j > i && A[i, j] > 0) {
      edge_df <- bind_rows(edge_df, data.frame(
        x    = station_coordinates$longitude[i],
        y    = station_coordinates$latitude[i],
        xend = station_coordinates$longitude[j],
        yend = station_coordinates$latitude[j]
      ))
    }
  }
}
cat(sprintf("edges drawn: %d (expect 33)\n", nrow(edge_df)))

p_graph <- ggplot() +
  geom_segment(data = edge_df, aes(x = x, y = y, xend = xend, yend = yend),
               linewidth = 0.5, alpha = 0.5) +
  geom_point(data = station_coordinates, aes(x = longitude, y = latitude),
             size = 3) +
  geom_text(data = station_coordinates,
            aes(x = longitude, y = latitude, label = station),
            nudge_y = 0.008, size = 3) +
  labs(title = "Geographic q-nearest-neighbor graph (q = 4)",
       x = "Longitude", y = "Latitude") +
  theme_minimal()

ggsave(file.path(OUT, "beijing_station_graph.png"), p_graph,
       width = 8, height = 6, dpi = 300)
cat("wrote", file.path(OUT, "beijing_station_graph.png"), "\n")
