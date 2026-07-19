idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "testdata/lpw__02_04_0001_gray64.bin"
)

set(BOUNDARYQAT_MODEL_PATH "${CMAKE_CURRENT_LIST_DIR}/models/boundaryqat_pupil.espdl")
esptool_py_flash_target_image(flash model 0x410000 "${BOUNDARYQAT_MODEL_PATH}")
