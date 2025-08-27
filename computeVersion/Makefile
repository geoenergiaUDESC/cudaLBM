include ../common.mk

EXECUTABLE = computeVersion
SOURCE = computeVersion.cu

default: clean $(EXECUTABLE)

$(EXECUTABLE):
	$(NVCXX) $(NVCXX_STANDARD) $(NVCXX_OPTFLAGS) $(NVCXX_MFLAGS) $(NVCXX_WFLAGS) $(NVCXX_COMPILER_WFLAGS) $(NVCXX_DFLAGS) $(SOURCE) -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

install: clean uninstall run $(EXECUTABLE)
	@ mkdir -p $(BIN_DIR)
	@ cp $(EXECUTABLE) $(BIN_DIR)/
	@ rm -f $(EXECUTABLE)

clean:
	@ rm -f $(EXECUTABLE)
	@ rm -f $(INCLUDE_DIR)/hardware.info

uninstall:
	@ rm -f $(BIN_DIR)/$(EXECUTABLE)