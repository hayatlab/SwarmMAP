library(readxl)
library(tidyverse)
library(stringr)

db <- readxl::read_xlsx("~/work/aachen/sclabel/utils/Cell_marker_Human.xlsx", sheet = 1)
db$tissue_type %>% unique %>% sort
db$cell_name
aa = db %>% filter(cell_name %>% stringr::str_ends("T cell")) %>%
  select(tissue_type)
aa$tissue_type %>% unique()
heart_markers <- db %>%
  # filter(tissue_type == "Heart")
  filter(tissue_type %in% c("Heart", "Heart muscle", "Ventricular and atrial",
                            "Ventricle", "Ventricular zone", "Lymph node")) ## alternative

# mi_celltype <- c("adipocyte", "prolif", "endothelial", "fibroblast", "lymphoid", "mast",
#                  "myeloid", "pericyte", "smooth muscle", "cardiomyocyte", "neuron")

heart_celltype <- c("adipocyte", "atrial cardiomyocyte", "cardiomyocyte", "cytoplasmic cardiomyocyte",
                    "endocardium", "endothelial", "epicardium", "fibroblast", "lymphatic",
                    "lymphoid", "mast", "mesothelial", "myeloid", "neuron", "pericyte", "prolif",
                    "smooth muscle")

db_celltypes <- heart_markers$cell_name %>% unique %>% stringr::str_to_lower() %>%
  str_remove("\ cell") %>% sort()
is_in_db <- heart_celltype %in% db_celltypes
matching_celltypes <- heart_celltype[is_in_db]
non_matching_celltypes <- heart_celltype[!is_in_db]
db_celltypes

## Change some names to have the correspondence: -----
# atrial cardiomyocyte -> atrial
# cytoplasmic cardiomyocyte -> cardiomyocyte
# endocardium -> endocardial
# epicardium -> epicardial
# lymphatic -> lymphatic endothelial
# lymphoid -> lymphocyte
# mast -> myeloid
# mesothelial -> NULL
# prolif[erating cells] -> NULL
heart_celltype <- heart_celltype %>%
  as.factor() %>%
  forcats::fct_recode(atrial = "atrial cardiomyocyte",
                      endocardial = "endocardium",
                      epicardial = "epicardium")
heart_celltype <- heart_celltype %>% as.character()
is_in_db <- heart_celltype %in% db_celltypes
matching_celltypes <- heart_celltype[is_in_db]
non_matching_celltypes <- heart_celltype[!is_in_db]

##
tmp <- heart_markers %>%
  mutate(cell_name = cell_name %>% stringr::str_to_lower() %>%
           str_remove("\ cell")) %>%
  filter(cell_name %in% heart_celltype[is_in_db]) %>%
  select(cell_name, Symbol) %>%
  group_by(cell_name)

tmp2 <- split(tmp$Symbol, tmp$cell_name)
tmp2 %>% str
"cardiomyocyte" %in% db_celltypes

tmp <- db$cell_name %>% unique()
tmp %>% View

diff_marker_symbol <- heart_markers %>%
  dplyr::select(marker, Symbol) %>%
  filter(marker != Symbol)
## conclusion: there are 28 genes that have different marker and Symbol names
## out of 303 genes. So the two naming conventions are  similar.

library(garnett)
garnett::train_cell_classifier()

library(org.Hs.eg.db)
columns(org.Hs.eg.db)
