include ../common.mk

EXECUTABLE = momentBasedD3Q19
SOURCE = LBM.cu

default: clean $(EXECUTABLE)

$(EXECUTABLE):
	$(NVCXX) $(NVCXX_FLAGS) $(SOURCE) -o $@

install: clean uninstall $(EXECUTABLE)
	@ mkdir -p $(BIN_DIR)
	@ cp $(EXECUTABLE) $(BIN_DIR)/
	@ rm -f $(EXECUTABLE)

clean:
	@ rm -f $(EXECUTABLE)

uninstall:
	@ rm -f $(BIN_DIR)/$(EXECUTABLE)