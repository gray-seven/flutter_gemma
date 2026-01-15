import 'package:flutter/foundation.dart';
import 'package:flutter_gemma/core/domain/model_source.dart';
import 'package:flutter_gemma/core/di/service_registry.dart';
import 'package:flutter_gemma/core/utils/file_name_utils.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:path/path.dart' as path;
import 'package:flutter_gemma/core/services/model_repository.dart' as repo;

/// Fluent builder for embedding model installation
///
/// Provides type-safe API for installing embedding models (requires model + tokenizer).
/// Automatically sets the installed model as the active embedding model.
///
/// Usage:
/// ```dart
/// await FlutterGemma.installEmbedder()
///   .modelFromNetwork('https://example.com/model.tflite', token: 'hf_...')
///   .tokenizerFromNetwork('https://example.com/tokenizer.model', token: 'hf_...')
///   .withModelProgress((p) => print('Model: $p%'))
///   .withTokenizerProgress((p) => print('Tokenizer: $p%'))
///   .install();
/// ```
class EmbeddingInstallationBuilder {
  ModelSource? _modelSource;
  ModelSource? _tokenizerSource;
  void Function(int progress)? _onModelProgress;
  void Function(int progress)? _onTokenizerProgress;
  CancelToken? _cancelToken;

  // === Model source setters ===

  /// Set model source from network URL (HTTP/HTTPS)
  EmbeddingInstallationBuilder modelFromNetwork(String url, {String? token}) {
    _modelSource = ModelSource.network(url, authToken: token);
    return this;
  }

  /// Set model source from Flutter asset
  EmbeddingInstallationBuilder modelFromAsset(String path) {
    _modelSource = ModelSource.asset(path);
    return this;
  }

  /// Set model source from bundled native resource
  EmbeddingInstallationBuilder modelFromBundled(String resourceName) {
    _modelSource = ModelSource.bundled(resourceName);
    return this;
  }

  /// Set model source from external file path
  EmbeddingInstallationBuilder modelFromFile(String path) {
    _modelSource = ModelSource.file(path);
    return this;
  }

  // === Tokenizer source setters ===

  /// Set tokenizer source from network URL (HTTP/HTTPS)
  EmbeddingInstallationBuilder tokenizerFromNetwork(String url, {String? token}) {
    _tokenizerSource = ModelSource.network(url, authToken: token);
    return this;
  }

  /// Set tokenizer source from Flutter asset
  EmbeddingInstallationBuilder tokenizerFromAsset(String path) {
    _tokenizerSource = ModelSource.asset(path);
    return this;
  }

  /// Set tokenizer source from bundled native resource
  EmbeddingInstallationBuilder tokenizerFromBundled(String resourceName) {
    _tokenizerSource = ModelSource.bundled(resourceName);
    return this;
  }

  /// Set tokenizer source from external file path
  EmbeddingInstallationBuilder tokenizerFromFile(String path) {
    _tokenizerSource = ModelSource.file(path);
    return this;
  }

  // === Progress callbacks ===

  /// Add model file progress callback
  EmbeddingInstallationBuilder withModelProgress(void Function(int progress) onProgress) {
    _onModelProgress = onProgress;
    return this;
  }

  /// Add tokenizer file progress callback
  EmbeddingInstallationBuilder withTokenizerProgress(void Function(int progress) onProgress) {
    _onTokenizerProgress = onProgress;
    return this;
  }

  /// Set cancellation token for this installation
  ///
  /// The same token will be used for both model and tokenizer downloads.
  ///
  /// Example:
  /// ```dart
  /// final cancelToken = CancelToken();
  ///
  /// final future = FlutterGemma.installEmbedder()
  ///   .modelFromNetwork(modelUrl)
  ///   .tokenizerFromNetwork(tokenizerUrl)
  ///   .withCancelToken(cancelToken)
  ///   .install();
  ///
  /// // Cancel from elsewhere
  /// cancelToken.cancel('User cancelled');
  /// ```
  EmbeddingInstallationBuilder withCancelToken(CancelToken cancelToken) {
    _cancelToken = cancelToken;
    return this;
  }

  /// Execute the installation and automatically set as active embedding model
  ///
  /// Returns [EmbeddingInstallation] with details about installed model.
  ///
  /// Throws:
  /// - [StateError] if model or tokenizer source not configured
  /// - [DownloadCancelledException] if cancelled via cancelToken
  /// - [Exception] on installation failure
  ///
  /// Note: This method is idempotent - calling install() on an already-installed
  /// model will skip download and just set it as active.
  Future<EmbeddingInstallation> install() async {
    // Check cancellation before starting
    _cancelToken?.throwIfCancelled();

    if (_modelSource == null || _tokenizerSource == null) {
      throw StateError(
        'Both model and tokenizer required. Use modelFromNetwork() and tokenizerFromNetwork().',
      );
    }

    // Create spec
    final modelFilename = _extractFilename(_modelSource!);
    final tokenizerFilename = _extractFilename(_tokenizerSource!);

    final spec = EmbeddingModelSpec(
      name: FileNameUtils.getBaseName(modelFilename),
      modelSource: _modelSource!,
      tokenizerSource: _tokenizerSource!,
      replacePolicy: ModelReplacePolicy.keep,
    );

    final registry = ServiceRegistry.instance;
    final repository = registry.modelRepository;

    // Check if both model and tokenizer are fully installed (metadata + file exists + correct size)
    final isModelInstalled = await _isFileFullyInstalled(modelFilename, repository, registry);
    final isTokenizerInstalled = await _isFileFullyInstalled(tokenizerFilename, repository, registry);

    if (isModelInstalled && isTokenizerInstalled) {
      debugPrint(
          'ℹ️  Embedding model already installed: $modelFilename + $tokenizerFilename (skipping download)');
    } else {
      final handlerRegistry = registry.sourceHandlerRegistry;

      // Install model file if not already installed
      if (!isModelInstalled) {
        debugPrint('📥 Installing embedding model...');
        final modelHandler = handlerRegistry.getHandler(_modelSource!);
        if (_onModelProgress != null) {
          await for (final progress in modelHandler!.installWithProgress(
            _modelSource!,
            cancelToken: _cancelToken,
          )) {
            _onModelProgress!(progress);
          }
        } else {
          await modelHandler!.install(
            _modelSource!,
            cancelToken: _cancelToken,
          );
        }
      } else {
        debugPrint('ℹ️  Embedding model file already installed: $modelFilename');
      }

      // Install tokenizer file if not already installed
      if (!isTokenizerInstalled) {
        debugPrint('📥 Installing tokenizer...');
        final tokenizerHandler = handlerRegistry.getHandler(_tokenizerSource!);
        if (_onTokenizerProgress != null) {
          await for (final progress in tokenizerHandler!.installWithProgress(
            _tokenizerSource!,
            cancelToken: _cancelToken,
          )) {
            _onTokenizerProgress!(progress);
          }
        } else {
          await tokenizerHandler!.install(
            _tokenizerSource!,
            cancelToken: _cancelToken,
          );
        }
      } else {
        debugPrint('ℹ️  Tokenizer file already installed: $tokenizerFilename');
      }
    }

    // AUTO-SET as active embedding model (even if already installed)
    final manager = FlutterGemmaPlugin.instance.modelManager;
    manager.setActiveModel(spec);

    debugPrint('✅ Embedding model installed and set as active: ${spec.name}');

    return EmbeddingInstallation(spec: spec);
  }

  String _extractFilename(ModelSource source) {
    return switch (source) {
      NetworkSource(:final url) => path.basename(Uri.parse(url).path),
      AssetSource(:final path) => path.split('/').last,
      BundledSource(:final resourceName) => resourceName,
      FileSource(:final path) => path.split('/').last,
    };
  }

  /// Validates if a file is fully installed (metadata exists + file exists + correct size)
  ///
  /// Returns true only if:
  /// 1. Model metadata exists in repository
  /// 2. File actually exists on disk/storage
  /// 3. File size matches the stored sizeBytes (or minimum size if stored size is 0)
  Future<bool> _isFileFullyInstalled(
    String filename,
    repo.ModelRepository repository,
    ServiceRegistry registry,
  ) async {
    try {
      // 1. Check if metadata exists in repository
      final hasMetadata = await repository.isInstalled(filename);
      if (!hasMetadata) {
        debugPrint('📋 No metadata found for: $filename');
        return false;
      }

      // 2. Load metadata to get stored size
      final modelInfo = await repository.loadModel(filename);
      if (modelInfo == null) {
        debugPrint('📋 Failed to load metadata for: $filename');
        return false;
      }

      // 3. Check if file actually exists
      final fileSystem = registry.fileSystemService;
      final filePath = await fileSystem.getTargetPath(filename);
      final fileExists = await fileSystem.fileExists(filePath);

      if (!fileExists) {
        debugPrint('📁 File not found on disk: $filename (metadata exists but file missing)');
        // Clean up orphaned metadata
        await repository.deleteModel(filename);
        return false;
      }

      // 4. Validate file size
      final actualSize = await fileSystem.getFileSize(filePath);
      final storedSize = modelInfo.sizeBytes;

      // For web, getFileSize returns -1 when size is unknown but file exists
      // In that case, we trust the file exists
      if (actualSize == -1) {
        debugPrint('✅ File exists (web, size unknown): $filename');
        return true;
      }

      // Check if file size matches stored size (with tolerance for small files)
      if (storedSize > 0 && actualSize < storedSize) {
        debugPrint(
          '⚠️  File size mismatch: $filename (expected: $storedSize bytes, actual: $actualSize bytes)',
        );
        // File is incomplete - clean up
        await repository.deleteModel(filename);
        return false;
      }

      // Also check minimum size based on file extension
      final extension = filename.contains('.') ? '.${filename.split('.').last}' : '';
      final minSize = FileNameUtils.getMinimumSize(extension);

      if (actualSize < minSize) {
        debugPrint(
          '⚠️  File too small: $filename (minimum: $minSize bytes, actual: $actualSize bytes)',
        );
        // File is too small (likely corrupted) - clean up
        await repository.deleteModel(filename);
        return false;
      }

      debugPrint('✅ File fully installed: $filename ($actualSize bytes)');
      return true;
    } catch (e) {
      debugPrint('❌ Error validating file: $filename - $e');
      return false;
    }
  }
}

/// Result of embedding model installation
class EmbeddingInstallation {
  final EmbeddingModelSpec spec;

  EmbeddingInstallation({required this.spec});

  /// Model ID (filename without extension)
  String get modelId => spec.name;
}
