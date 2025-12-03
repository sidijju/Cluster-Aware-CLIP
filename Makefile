SETUP_SCRIPT=scripts/setup_cpu.sh
SETUP_GPU_SCRIPT=scripts/setup_gpu.sh
COCO_SCRIPT=scripts/download_coco.sh

all: setup

setup:
	@echo "🔧 Running setup..."
	chmod +x $(SETUP_SCRIPT)
	./$(SETUP_SCRIPT)

setup_gpu:
	@echo "🔧 Running setup for GPU..."
	chmod +x $(SETUP_GPU_SCRIPT)
	./$(SETUP_GPU_SCRIPT)

coco:
	@echo "⬇️  Downloading MS COCO..."
	chmod +x $(COCO_SCRIPT)
	./$(COCO_SCRIPT)

# TODO: future datasets

clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf tmp/*
	rm -rf .venv

.PHONY: all setup coco imagenet clean
