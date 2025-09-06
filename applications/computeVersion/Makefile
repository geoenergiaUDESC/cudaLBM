include ../../common.mk

EXECUTABLE = computeVersion
SOURCE = computeVersion.cu

default: clean $(EXECUTABLE)

$(EXECUTABLE):
	$(NVCXX) $(NVCXX_STANDARD) $(NVCXX_OPTFLAGS) $(NVCXX_MFLAGS) $(NVCXX_WFLAGS) $(NVCXX_COMPILER_WFLAGS) $(NVCXX_DFLAGS) $(SOURCE) -o $@

run: $(EXECUTABLE)
	@ ./$(EXECUTABLE)

install: clean uninstall run $(EXECUTABLE)
	@ mkdir -p $(CUDALBM_BIN_DIR)
	@ mv $(EXECUTABLE) $(CUDALBM_BIN_DIR)/
	@ mv hardware.info $(CUDALBM_INCLUDE_DIR)/
	@ rm -f $(EXECUTABLE)

clean:
	@ rm -f $(EXECUTABLE)
	@ rm -f $(CUDALBM_INCLUDE_DIR)/hardware.info

uninstall:
	@ rm -f $(CUDALBM_BIN_DIR)/$(EXECUTABLE)