Detach chunk adapter and disconnect NodeMap for Harvester instances.
Update S3Manager to use list_buckets method
Refactor error reporting logic for clarity and efficiency
Add profiling to server loop
Use MetricUnit for time units
Fix import of `collections.abc` based on Python version
Simplify `testStore` by removing redundant assertion.

Remove debugging print statements from `unescape` function.
Add traceback printing for failed module loading when debug level is high
Update copyright and version information
Update paths and timespan for vb suite
Add option to disable automatic cross-seeding and explicit label for it (#<issue number if applicable>)
Improve logging documentation for MAGICC versions.
Fix renormalization handling in spectrum parser
Refactor: Remove unnecessary check for FIFO in chunk consumer
Add minimum BPDU length and register type for new BPDU subtypes.
Enable digital pin reporting on port initialization
Fix: Insert host object correctly in SortedHostCollection
Remove redundant `legislative_input` initialization
Handle potential errors when retrieving vault information from Braintree.
Refine GO DAG version string for clarity
Add color definitions with resource link
Use classmethod for b58decode and update Address instantiation
Fix: Handle empty YAML blocks in doc metadata
Fix test case for ToolboxHtmlToolsTest
Add `last_modified` parameter to `send_file` function
Refactor: Change heartbeat failure logging from exception to debug

Add probe service and constants
Add grouping functionality to `list_metafeatures` method
Fix: Regex for CRTF header matching
Ensure ControlConnection methods use a valid connection before calling internal functions
Fix mkdir_p to avoid creating directories in the same place as parent directories
Add 'release' to sphinx dependency trigger
Bump version to 1.0.2
Add Darwin platform support
Ensure client disconnection/reconnection in network GUI test
Bump version to 4.0.2

Add tests for index dot slices
Move `SoS_Exporter` definition to `sos/jupyter/kernel.py`
Add data files for default broker and client configurations
Add flake8 ignore comment
Set MACOSX_DEPLOYMENT_TARGET for OSX builds
Add optional type hint for `window_shell` in Session.__init__
Update wheel dependency to version 0.30
Use 'Agg' backend for matplotlib in syncpr tests
Fix potential issue with resetting parsed streams in TTL parser.
Add support for AWS profiles
Add captureoutput tag for capturing template output into a variable
Disable closing memcached connection on teardown
Bump version to 1.4.0 (dev)
Add blur option to heatmap
Add test for code blocks and syntax boundaries
Add Mizuki voice resource
Add Read the Docs support
Use Dask delayed for NinjoTIFF saving
Refactor: Simplify Git clone logic
Increase thumbnail size
Ignore non-comment lines from engine during communication
Add `bool_only` parameter to DataFrame.count() for filtering boolean columns.
Fix Action.to_rest() to use dp object and clarify action logic.
Remove retry logic from k8s tests
Fix: Domain deletion and retrieval tests
Fix handling of read-only fields in browseable API for non-existent resources
Add root_url parameter to JSHost
Simplify `open` method by using keyword arguments directly.
Fix: Use partials in Renderer instantiation
Refactor: Thoroughly clear homely modules before setting up a test HOME directory.
Refactor: Use Connection class for client connections
Handle property definitions in _impl phase
Add support for ellipse regions in ShapeList output.
Add assertion for empty response content in HEAD requests
Refactor: Improve efficiency of fixing VCF headers
Pass config to message processor constructor
Simplify boolean_and_simplifier by handling single-argument case.
Refactor: Move Board import within Session class
Handle potential TypeError when parsing temporal attributes
Refine source path cleaning for gcloud projects

Add tests for allowed capabilities and DescribeProcess requests
Allow specifying number of trainable ResNet layers in `resnet_fpn_backbone` function. (#XXXX)
Handle datetime objects correctly in `Calendar.is_holiday()`
Add test for reading empty arrays with various dtypes (#41241)
Fix permission check logic for yAuth decorators
Refactor: Improve gender matching logic with `get_token_features` function
Fix handling of whole column/row selection in ArrayView
Skip test_fuse_projections
Add new properties to StageDescription class
Refactor: Remove debug print statements from SqlaTable ordering
Add shortcut methods for `call_` and `call_mut_`
Remove Markdown support from documentation
Update droplet IDs to UIDs in job execution flow.
Handle potential OSError when getting file size
Refactor test names for consistency.
Add test for sstable_size_in_mb option
Increase benchmark iterations to 100,000
Fix: Ensure intopt is called for models with integer or binary variables.
Fix: Skip doctest for normalize_arxiv_category function.
Fix handle version increment logic and use signals instead of deprecated method.
Set _highest_parent_ for all parameters
Add script for starting validator with delay
Fix typo in HPC configuration template
Fix: Plot potentials instead of densities
Remove unnecessary exception handling in pkgrepo state
Use list for urlpatterns instead of patterns.
Fix HTML parsing by removing unnecessary boolean flag
Add whitelist support to _process_Stmt method
Bump version to 0.3.3 and update development status to Production/Stable
Fix source label spacing on macOS
Change upscale method from pypdf3 to pdfrw
Add project metadata and classifiers
Bump version to 1.8.6
Add debug logging for SRPM resolution
Bump version to 0.3.1
Refactor: Simplify append check in file.append state
Update ipyvolume version to 0.5.0 and remove beta tag from JS version.
Use run_sql for bibwords search
Enhance tick limit handling in HolodeckEnvironment
Clarify comparison between approx and other methods
Refactor: Set 'tty' default based on SSH user
Enable stop button and set algorithm stop flag when run is triggered.
Only use managed API if not running locally.
Fix country flag image replacement logic
Fix: Use key directly instead of Variable in UsersAvatarObjectNode.render()
Add a new method 'foo' to the Motor class.
Fix Source detachment handling for Stripe API v2
Fix: Use not kwargs['block_signing_key'] instead of empty check
Fix encoding of PEP517_BUILD_BACKEND for Python 2 compatibility.
Add __hash__ method to ContextOptions (temporary)
Fix: Handle file cache retrieval and closing properly
Add network variable to etrago function
Use absolute path for git config file in version release test
Check for existence of install config file before loading
Rename ASN1_raw to ASN1_force and handle nested ASN1_Objects in encoding.
Fix: Call init_app method on LoginManager instance
Add charset-normalizer as a dependency
Remove required constraint from KotlincPluginIdField
Add Thermo-specific processing for MZML files
Use DateTime(timezone=False) for created_at columns
Fix: Ensure comma is added after first argument in JSON object.
Fix: Use underscore for local modules in SSH state
Add t.Write() to close file after data transfer.
Resolve issue #88: Set TensorFlow session in KerasRunner.
Handle duplicate waypoints and request next waypoint
Update invoice ledger line date format to ISO 8601.
Exclude MixtureSameFamily from PYTREE_BLOCKLIST due to performance issues.
Update schneier command to use requests library and HTMLParser
Add support for parsing Googlebot, MSNBot, Yahoo! and Ask Jeeves user agents.
Fix verbose flag handling in VerboseOptimization
Fix current electrode ordering in EIT40 importer
Fix: Use ReceiptPDFView instead of ReceiptPDFDisplayView in test_views.
Add warning if name, email and username are missing in enrichment
Fix: Handle Arista login banner
Refactor: Use `significant_figures` instead of `float_prec` for formatting floats.
Bump version to 2.0.0
Refactor status.pid test to check cli_master process
Add function to find nearest expected polar angle for improved model selection.
Clarify environment variable usage in BaseTool.
Add 'wafer.compare' app to INSTALLED_APPS
Update seed.prep_bootstrap to return a tuple
Add optional parameters to IntrospectionQuery
Yield entire hit document instead of just '_source' in search results.
Bump version to 2.0.2 and ensure README encoding
Fix: Handle missing image files when using a custom URL prefix.
Add support for indexing Color objects and improve type checking in Shape.__init__
Use sharedFileName directly in UUID generation for file IDs
Improve timeout handling in HardwareSource playback tests.
Use f-strings for error messages in int_to_roman function.
Refactor: Move executor instantiation inside execute_action
Bump setuptools requirement to >= 3.7.0
Fix format string parsing for signed and unsigned integers\n\nThis commit fixes a bug in the format string parser by correctly handling both signed and unsigned integers. Previously, only signed integers were handled correctly. This commit also updates the `sim_atoi_inner` function to support more format specifiers.
Bump version to 0.8.1
Allow specifying log directory for daemonized processes
Use _lattice attributes instead of lattice for primitive structure comparisons.
Fix: Clarify literalinclude usage for example DAGs
Fix import order in admin_interface_tags.py
Remove unnecessary comment
Remove package install/removal timeouts for safer execution.\n\nThis change disables the global timeout during package installation and removal operations. This addresses potential issues where long-running post-installation scripts could cause timeouts, particularly for packages like ceph-selinux.
Update BioGrid dataset version handling
Refactor: Remove clock difference tolerance check
Print attribute relationship test statistics
Fix: Use `>` instead of `> 0` for checking  `self.per_experiment_nb_cpus`
Remove unused `call` function from subprocess imports
Improve performance test by adjusting entity creation and beating logic.\n\n- Increase the number of entities created for testing from 4 to 5.\n- Modify the 'num_beats' calculation to use `floor(10 ** i)` instead of `(10 ** i)`, ensuring a more consistent and controlled beat rate.
Ensure JSONFormField only validates strings
Fix: Correctly extract path from touch output
Fix linking of loop expressions and conditions in function parsing.
Fix classpath resolution in ZincUtils tests\n\nTest cases now correctly handle classpath elements relative to the build root.
Add twine to dev dependencies for publishing packages
Use README.rst instead of README for long description
Retry connection attempts for USBMux with exponential backoff.
Update `Mapping` import to use `six.moves.collections_abc`
Bump version to 0.0.6
Fix include file handling
Bump version to 0.14.1
Remove unused matplotlib imports
Only include environment pairs with '=' in them
Fix: Update auto_offset_reset behavior and error handling
Relaxed check on unused field in PrimaryVolumeDescriptor (PVD)
Add plugin names from _active_preset to __dir__ results

Bump version to 0.1.17

Ensure log file is writable before instantiation.
Refactor: Remove redundant tearDown method in TestSetupCLI
Use built-in StringIO instead of cStringIO for compatibility.
Fix initialization based on QT_API

Fix: Correct callback type for game status function
Use sets for comparing unique keys in probability_map.\n\nThis ensures that the order of elements is not considered during comparison, which is more robust and aligns with the expected behavior.
Refactor: Simplify TurkeyExperiment blocking logic
Import `wraps` from boltons.funcutils
Bump wcxf dependency to >=1.4.6
Renamed `costType` to `cost_type` in ChampionSpellData._renamed
Clarify help text for --absolute-time option
Add retrieval and printing of repository properties
Fix: Add missing call to get_ad_metadata
Use time.monotonic() for timing in retry logic
Ensure sigma is non-negative in NormalMessage initialization.
Add visualization of qubit layout using NetworkX and Matplotlib.
Import statements reordered for clarity
Ensure integer dtype for random data in merge tests
Change log levels for informational messages from INFO to DEBUG.
Fix GPS heading calculation in FlightGear integration
Refactor: Move vtk type mapping to _vtk_common module.
Fix: Minor formatting issue in logger output
Refactor CFScene scaling and offset calculation
Update copyright year and DockerRegistry docstring
Bump version to 0.0.0
Fix JPEG encoding for multi-channel images.
Simplify package structure in setup.py
Fix duplicate sample name handling in verifybamid module
Refactor: Extract language code from _PageIndex
Refactor: Improve child controller removal logic
Calculate tax rate for InvoiceItems based on taxes and total amount
Fix Image model saving logic
Add security warnings about running testapp in production
Fix salt-cloud command argument formatting
Implement order retrieval, withdraw functionality and deposit address query
Store last stdout and stderr for each command execution
Add edx package to setup.py
Fix CSRF error handling to use 403 status code instead of 400.
Refine intermittent commenter task schedules for more precision.
Swapped polite error messages for more... colorful ones.
Add tests for bounding box overlap slices and shape.
Refactor: Use f-strings for logging in download methods
Refactor: Release file lock by closing fd and resetting it
Ensure at least one task is defined in build configuration
Fix: Remove temporary unzipped file even if it was not modified.
Convert set of non-followers to list for unfollowing.
Fix path generation and instance retrieval for mptt URLs.
Add filter parameters to CouchDBRiver serialization
Bump version to 0.2.1
Bump version to 2.1.3
Add error handling for ProgrammingErrors during query execution
Refactor: Simplify encoding of unsafe characters in tasks.
Bump version to 0.1.0a2
Add OCSP key generation and CRL caching
Execute the state machine until a specific state. This state won't be executed. This is an asynchronous task
Fix test_fork_exec by using sleep 0 instead of 1
Refactor: Rename `vertical_effective_stress` to `vert_eff_stress` and mark original as deprecated.
Refactor: Use custom exception for entrypoint waiter timeout
Handle positional arguments in Callable
Allow empty SQL passwords for read-only access
Add test for container creation with CPU RT options
Use six.assertRaisesRegex for ValidationService test
Renamed Markov class to Markov
Add result property and lazy loading
Improve hazard curve data insertion efficiency by using a bulk inserter with caching.
Avoid PK-only optimization for nested hyperlinked fields
Update macOS tag in manifest validator.
Add tests for isinstance() with int(x) as argument
Update impact file handling to use impactLayer
Allow specifying exit command for 'exit_enable_mode' function
Make stream thread daemon status configurable
Improve legend documentation for Axes class
Fix: Use '-x enum' flag with simpledeps in test_pydeps_colors().
Add support for 'singlehtml' builder in asset copying.

Add __unicode__ method to KeyPublicationStatus model for better string representation.
Bump version to 0.3.7
Improve error/warning handling in ROOT integration
Update Gateway log messages to include host address
Refactor command handling for quieter error messages
Fix indentation in get_current_user method
Enable multiple commits for autosynth
Remove unused loop variable from publish method
Update GIT_ATTRIBUTES for dataset tasks
Improve bar spacing in chart
Add backports and libfuturize packages.
Bump version to 1.3.0
Handle potential git describe error gracefully in setup.py
Deprecate `duration` and `extent` methods
Allow API calls without certificate verification
Add test for random control frame handling
Fix: Open output directory using QUrl.fromLocalFile
Update handling of classroom usage in course settings\n\nThis change ensures that classrooms are properly handled when updating course settings. It now checks for 'use_classrooms' setting and deletes any existing aggregations if it changes.
Use utf-8 encoding when reading Jupyter notebooks from Pandoc
Add test for creating a TorConfig object from a protocol
Refactor: Rename `_update_recurring_next_date` method
Add a minimum status column to RunSetResult if all custom columns are excluded
Use yield from for recursive listing of files.
Change router yard prefix from 'com' to 'master'
Add logging for missing grid mapping variable detection #1234
Refine API documentation
Fix regex pattern in zypper module
Add ContentItemRelation to ContentsEntryMixin for better admin navigation.
Fix: Return correct list of states in get_all_states method.
Fix: Display mouse position without offsetting pixel coordinates
Remove Property printing logic from SudsObject.py
Add test for SNMPGet failure after shutdown
Allow passing dev and npm flags to npm install in install_web function
Fix: Handle null result in runner.py\n\nThis commit fixes a potential issue where `run_stats` could have an empty `result` value.\n\nThe original code only checked for `not result`, which would not catch cases where `result` was explicitly set to None. This commit changes the check to `if result is None:` to ensure that we correctly handle both scenarios.

Fix updating particle positions in `update_state_global`
Improve error message in parse_expression function
Add map editor functionality
Add __unicode__ method to CssBackgroundAbstractBase
Add support for 2D WCS projections in wcs_to_axes.
Remove include_package_data
Update HealthEventFilter to include 'upcoming' status
Update Application version support in client_facades
Refactor: Improve documentation links for SegmentationPurityCoverageFMeasure
Increase timeout for `test_attach_stream_and_cancel` to 10 seconds
Add `statistics` import to titled imports test.
Use OrderedDict for path, object, namespace mappings in SphinxMapperBase\n\nThis improves consistency and ensures that the order of elements is preserved.
Use f-strings for string formatting in ExternalTable class
Update license information in setup.py
Ensure CLI scripts are rewritten for each test run
Add Calamari and RadosGW logs to Ceph plugin
Increase retries for DataProc job retrieval.
Fix: Return symbol from find_market when it's a dictionary
Remove unused subHMM models
Update tdf_init index for datetime conversion in test_magicc_data_append
Add test for retrieving full path of very large file\n\nThis commit adds a new test case to verify that the function `iso.full_path_from_dirrecord` correctly returns the full path of a very large file added to an ISO image.
Bump snorkel-metal dependency to >=0.1.7
Fix dependency resolution for different pip versions
Allow fixtures with scope 'class' and 'module' in pytest collection modification
Add tests for larger numbers in Portuguese-Brazil

Fix known hosts file generation for non-default ports
Add support for RHEL and Fedora distros
Allow user-defined models in BayesianOptimization
Refactor: Handle min_mag input more robustly.
Add tests for negative temporal parameters in split function.
Use README.md as long description if pypandoc is missing
Set API version in AppObjBaseClass
Run tests with hgroup ignored by default when following W3C specification
Remove default reactions directory and parsing logic
Add default_image_config parameter to CompositeBase.__init__
Remove unnecessary header authentication parameter from Twitter API call.
Fix indentation in waterfall.py
Improve remote user authentication logic.
Allow specifying extraction directory for ZipExtractor
Update impact function names for consistency and clarity (#1234)
Refactor: Log sensitive data in POST requests with XXXXXXXX
Rename `dictionary` to `f2p_dictionary` for clarity
Test xatop: Ensure deleteFromStore respects order and limit with UPDATE_DELETE_LIMIT enabled.
Remove ordering from ArticleAdmin
Fix: Check for existence of FLAX weights file.
Fix: Use ARGS.outputDir instead of ARGS.output for identifier output directory
Improve task queue logging and clarity
Refactor: Use testbed init_blobstore_stub instead of registering stub directly
Bump version to 0.1.1
Bump swauth version to 1.0.9dev
Display memory in MiB instead of GB
Remove Python 2.7 specific dependency
Fix websocket endpoint handling
Add product environment requirement for deploy task
Use dedicated function for long description
Refactor: Move element_methods docstring for clarity
Fix circuit extension command formatting and handle string router IDs
Add Xliff as a supported file type
Refactor: Remove unnecessary keyword arguments from functionTagProcessor constructor
Remove CancelAndReinstate permission from BikaGenerator
Add os module for file system operations
Remove unnecessary close call in plotRealImg function
Update BART model to share embeddings between encoder and decoder
Rename `insp_files_dict` to `inspirals_outs_dict` for consistency
Clarify ref type for triggering builds
Update license specification in setup.py
Fix: Truncate explanation count in pytest output for CI builds
Only assert absolute path for non-Windows platforms
Update betamax dependency range for tests
Handle two-factor authentication during login
Add functionality to create new addresses in a wallet
Remove test_check_and_raise due to redundancy
Rename `_normalization` to `normalization` and update setter logic.
Restrict interpreter versions tested on MacOS for performance reasons.
Refactor: Improve efficiency of EOS mixing methods
Fix: Handle empty entity JSON data
Remove debug logs for container objects in Kubernetes check.

Fix flash writing calculation for larger images.
Refactor: Simplify auto-indentation formatting in PercentStyleMultiline.
Refactor: Remove hidden attributes from CRUDResponse flatten method
Improve Pipe repr by using function name for LambdaComponent
Bump version to 0.21.0
Save memory by shutting down parallel processing after composite source model filtering.
Use Dark2 color scale for lineage visualization
Fix calculation of timezone offset in FixedOffsetTimezone
Fix removal of directories on Windows
Upgrade LoginAccount version 1 to 2, handle password decoding
Improve endpoint parsing in metadata.py
Handle potential errors when fetching server info from the API.
Prompt user for app name if not provided on command line
Fix PlayerManager to handle different entity types
Refactor: Use dictionary for connection parameters
Fix ownership of remote repository working directory
Fix: Handle empty files and missing cluster data gracefully\n\n- Return 0 if file reading fails or no clusters are found.\n- Add checks to handle empty input files and return 0 accordingly.
Add support for 'linux' platform in _GetNativeEolStyle
Refactor: Move `verify_certs` function outside class

Add handling for 'phenotype' MIM type
Fix negative binomial log prob autovectorization (#147743999)
Add wait for ready state complete after sending keys if setting is enabled
Add support for 1D input points in CDR.__call__
Refactor: Move ping option and waittime/count options into separate method calls
Fix: Handle invalid input and reset base prompt after command execution
Update test data URL for MDF tests
Use taggit_selectize instead of taggit_autosuggest for TaggableManager.
Refactor: Parse dumped grids as single objects.
Add field setting after argument sorting
Add LD_LIBRARY_PATH for Intel MKL on Stampede cluster.\n\nThis commit adds necessary environment variables for running Python scripts with Intel MKL on the xsede.stampede cluster.
Handle unsupported target types during Mbed board discovery
Fix: Use == for comparison in push_and_tag_images()
Fix typo in XLM-R tokenization description

Fix Spectrum filtering by using *= for in-place operations
Add integration tests for AdsApi and improve test robustness
Add docstring example for tag_ner function
Add django-oauth-toolkit and clean up dependencies
Fix: Ensure requests are saved before checking homepage content
Bump version to 0.1.3
Assert rundir is configured in get_pidpath function.
Update M4 lexer name consistency
Refactor: Extract channel name directly from URL for JustinTV plugin.
Fix: Skip string type validation for specific fields
Add support for grouping by multiple columns.
Add connection state tracking for FTP target
Handle potential `fastavro` dependency absence in SchemaRegistryError module
Add test for unreleased items going into unreleased release
Bump version to 0.8.1
Fix: Concatenate a structured grid to this grid
Improve TracService URI formatting
Add close method to DaskResource and ensure proper cleanup.
Add Python 3.6 and 3.6 notebook support
Fix: Handle missing 'stl' module in STL reader
Add as_dict method to ConfigModel for creating a dictionary representation of the config.\n\nThis allows easier access and manipulation of the config data.
Add psycopg2 as a dependency
Refactor: Improve timezone handling in template expansion
Ensure URLs and body are encoded as bytestrings for compatibility with Python 2.7.
Renamed `BUILDS_STOP` task to `BUILD_JOBS_STOP`
Fix: Use appropriate file descriptor for safe output.\n\nPreviously, `sys.stdout` was used regardless of the mode. This change ensures that `sys.stdout.buffer` is used when the mode is not 'w' to avoid potential issues with encoding.
Use functools32.lru_cache for Python 2.7 compatibility.

Add support for specifying users in notifications
Refactor: Improve omit function handling of non-dict input
Add RegionProposal and ProposalPostProcessor to scala class extraction.
Use 'generate' instead of 'setgenerate' for regtest daemon service.
Refactor: Set DEBUG setting based on command options
Bump version to 2.3.0.dev0
Add return statement to randomizer_bin_und function

Remove unused unittest import
Add `line_break_count` parameter to `write_line_list` for more control over line breaks
Use `Path` directly from 'path' module
Add HazardMapDBReader class for reading hazard map data from the database.
Fix: Use bytes for authentication messages
Remove debug logging from urlencoded_processor
Simplify PandasQueryCompilerView constructor and remove redundant reindexing logic.
Refactor: Use renku version for commit metadata
Add tests for query format with whiteboard and fedora-review flag
Remove unnecessary print statement and logging info
Use display layers to find data index for complex display type chooser
Add additional parameters to customized_warning function
Refine split_all logic for composite source models
Use unittest.mock or mock depending on environment
Require datacenter argument for `info` command
Switch from ChromeDownloader to WgetDownloader for urlretrieve
Add tests for DeVosSolarcell calculators with zero bandgap
Refactor: Improve docstring for debug decorator and add missing imports
Print statement added for debugging in increaseamount_assemble_one_step function
Fix HTML encoding detection in minification

Fix MQTT client disconnection by ensuring all data is sent and received.
Add safe string marking for email preview link in EmailTemplateAdmin
Add support for arguments and keyword arguments when registering handlers.
Remove object_class from BaseModelResource Meta
Log Saltfile loading information
Add support for list addition with Force objects
Refactor: Move view name and description retrieval
Implement Hessenberg decomposition via Householder transformations
Ensure consistent UTF-8 encoding for object uploads across Python versions
Fix: Delete domain doesn't remove others
Refactor: Move low-level function lookup to last step
Fix SSLWantReadError handling in HttpUA
Set Windows taskbar icon for conda-manager app
Fix: Explicitly specify UTF-8 encoding for file reading
Fix: Skip Foreign Key columns correctly in ModelConverter
Refactor: Simplify ReaderEventNotification encoding
Add per-file output for Pylint reports
Fix: Reverse file ID and path in updateFile
Refine listener cancellation logic
Add created_at to StatementAdmin list display and filter
Add error logging when aborting connection without a transport.
Add a function to generate new random tokens
Add `cross` parameter to `spectrogram` method
Add viz module
Add bs4 dependency for Itella parcel tracking module.
Add more informative error messages for failed D4S2 requests.
Ensure build process starts with pending tasks and sets finished flag.
Add randomness and hashing to minion config file paths
Only capture Sentry errors for non-ignorable 404s
Refactor: Move lldp execution logic to _exec_lldp
Remove debugging print statement
Fix: Handle `None` value for notification properly
Fix: Generate public key if it doesn't exist
Add get_items method to Cart class
Fix git version parsing and PR regex
Improve type hinting for `excluded_entities` parameter in `bulk_lookup_rdap` function.
Refactor: Use query for get_statement_jsons_from_hashes

Use image name instead of 'self.image' when tagging and pushing
Improve agent shutdown by using SIGKILL instead of SIGINT
Add Content-Type header for DELETE requests in AioHttpInterface
Fix regex patterns for notebook address and directory
Improve output formatting for healthcare access details
Add test module for TensorFlow
Refine regex for string matching in completer.
Add unicode_literals import for Python 3 compatibility
Adjust figure width in ethnicity check plots

Fix: Use `args` instead of `arg_list` in `test_exception_invalid_arg`
Fix minion_publish method to correctly access load dictionary
Fix memory calculation for LSF batch system
Fix CRL creation logic for serial numbers
Use `abc.Iterable` instead of `collections.Iterable` for type checking.
Fix compatibility check for ExceptionGroup.
Fix: Handle resolving nonexistent relationships gracefully
Handle TotalConsumption for legacy protocol gracefully.
Configure logging level for requests in Nova cloud provider
Fix: Import helperservices correctly
Remove unnecessary Optional from warns() function signature
Fix BinaryQuadraticModelStructureError message
Add beautifulsoup4 dependency
Initialize `req` variable in WebApp
Fix type mismatch in evaluation logic
Update Lambda runtime options to include Python 3.9
Add input functionality for tenant schema management commands
Fix: Use correct session instance in Resource class
Disable HTML output in QtConsole\n\nQtconsole doesn't report its line width properly and behaves unexpectedly when displaying HTML tables that exceed the window size. This commit disables HTML output for QtConsole to avoid these issues.
Fix: Ensure correct working directory for themed_tk.\n\nThis commit ensures that the themed_tk module is using the correct working directory when loading themes.  Previously, it was changing the working directory and not restoring it afterwards, which could lead to issues with other modules.
Require API key during initialization.
Fix: Use __path__ for static file mounting from packages
Fix type handling for ListSerializer fields and write-only serializers
Refresh selected state machine and clean loaded libraries
Fix typo in MultiQC output directory name.
Refactor: Use posix paths for remote transfers in StageCache
Use virtualchain library for hashing and key generation
Remove compatibility wrapper for old PyWin32 errors.
Add `medians` method to MultiDict
Test: Switch from testing Schmidt normality to Schmidt seminormality.

Redirect LXC subprocess stderr to stdout
Import statements reorganized for clarity
Add `force_color` option to `get_printer` function.
Remove redundant sys.version check in TestTestTest
Refactor: Remove initialization from shell update and simplify code.
Refactor softmax implementation for improved efficiency
Fix: Use 'is not None' for null checks in deci2sexa function.
Use json.loads instead of json.load for parsing Wikidata response
Fix issue with setting temperature to 0. Disable nPsisAuto and set nPsis to 1 when temperature is 0.
Add SSL support and default `all_ok` to False for interactive server
Fix task management for DisaggHazardCalculator
Fix: Remove trailing whitespace from rendered list items
Allow `const_module` and `instruments` to accept modules and lists respectively for improved flexibility.
Reduce maximum pool size to 50
Add test for clearing surroundings
Reduce MAX_NAME_LENGTH for better Travis CI compatibility.
Reduce wait time for subprocess closure before restarting Spyder.
Bump version to 0.4.12
Strip extra newline characters from headers in HTTP1Connection
Refactor: Use XPath for finding links by partial text and full text
Ensure DictInterface handles various data types correctly

Rename `id2info` to `id2source` for clarity
Add host.docker.internal for Linux environments
Refactor: Use provided tasks or recipe for process list
Fix: Return a tuple with an error message for unsupported platforms in win_service module.
Add sidebars to documentation
Fix regex pattern in bootstrap_oembed
Add internal error handling for fuzzy hashing
Fix typo: 'operationi' -> 'operations'
Add matplotlib, wxPython and PIL as dependencies.
Fix: Ensure sets in DisaggregationBinaryMatrixXMLWriter have at least one element
Update serial dependency to pyserial
Simplify TIMEZONE_IDS generation
Refactor event creation logic

Use setUpClass and tearDownClass for API server setup and teardown.
Refactor: Improve readability of Checkpoint instantiation.
Use frozenset for storage parameter in get_user_data function
Fix: Ensure correct class assignment when cloning MultilingualQuerySet
Add tests for different branches and fossil not installed
Fix autoname for Viaplay clips
Refactor InterpolationTask and LinearInterpolation: Use consistent naming conventions for libraries and parameters.
Add placeholders for mongo profile and ss transactions in metrics report.
Fix: Handle cases where value type differs from current value in isChanged() method.
Add DocBookBodySubNode
Fix: Ensure transferred files have .gz extension
Add functionality to store sent emails in database.
Add 'Show Map' and 'Show HUD' options to MAVProxy console menu
Add tests for shell command sleep before execution.
Refactor: Test for getting resource list
Add deprecated package name and update setup.py
Add axis labels and legend to timeseries plot.
Update database connection string
Use getattr to retrieve fill_value from NDArrayField
Fix: Remove PrintSpace dependency from Workspace.
Remove debug logs for House Status Monitor events
Remove unused urllib.parse import
Fix Docker connection handling in glances_docker plugin
Remove HipChat provider
Add fallback for filelock module in ICS NeoVi backend
Fix nested value lookup in YamlTestCase
Print call arguments and error output instead of logging
Add focus to canvas after sending command
Add Numpy array support conditionally
Sort aliases in SymbolIndex for consistent ordering
Use a set instead of list for WEBHOOK_EVENT_TYPES
Fix: Remove trailing '*' from translated protein sequence
Use all available CPU cores for parallel processing in `scan_dir`.
Set tooltip background color

Remove rewriting of maven_metadata_path
Permit certain punctuation characters within tokens.
Fix: Argument parsing for action option
Add models_dir function
Lowercase package directory name for consistency.
Handle return value change in find_and_modify for MongoDB 2.1.2 and above.
Clarify UnknownExtension exception message
Fix sorting of chained select choices
Add --websocket-port only if --connect-existing is not specified.
Add functionality to OrderableManager for accessing previous ordered objects
Remove support for Python 2.6 BZ2File context manager protocol
Mock mkdir calls in test_blog.py
Ensure consistent return data structure for local and publish calls
Fix: Handle None return value correctly in test_stats_service.
Refactor: Use py_modules and package_dir instead of find_packages
Remove strict mode from RequestSchema in tests

Add more functions to function mapping.
Fix: Ensure `_git` object is baked after setting `_tty_out`.
Add CRUD event creation for M2M changes
Refactor: Improve Composition equality check
Fix: Properly format public key for verification
Use sampling with replacement for all parameters if at least one is a distribution.
Fix: Raise error for undefined terminals in grammar
Improve file export output for zip files
Use unicode literals for newline characters.
Use `items()` instead of `iteritems()` for Python 3 compatibility
Refactor: Transform accent handling in functional style
Add Red Hat specific upload configuration
Improve DNS server ordering logic in win_dns_client state
Reset task_try_count on each new Task instance.

Use broker connection URI for conninfo in startup info
Refine stats display: Show 'No stats found.' instead of an error
Update model.step to include state and dones.
Fix handling of environment variables and paths on Windows
Enforce -t/--time as a required argument and ensure mutually exclusive groups are always used.
Skip process exit tests on non-POSIX systems and Python 3 (Twisted limitation)
Remove 'index' column from aggregated data for consistency
Add .gitignore for s3conf directory
Add support for light and bold text styles based on weight
Bump version to 2.2.1

Fix Hz reporting on BeagleBone by retrieving advertised frequency.
Update ARAnalysesField to modify existing AR specs with new form values
Refactor: Open HID device using .open_path()
Add tests for argument parser
Fix endianness in RadioTap.len field
Add synchronous execution support
Add dependencies ujson and sh
Fix filesystem context menu to correctly handle desktop file associations.
Add support for shared memory in StimelaJob
Bump version to 0.5.0
Refactor: Improve parsing of IO samples in ZigBee class
Refactor: Use Infomap instance methods for network creation and execution.\n\nThis change simplifies the code by using `myInfomap.network()` to create the network and `myInfomap.run()` to execute the Infomap algorithm. It also updates the output formatting for clarity.
Refactor: Remove unnecessary configuration parameters from `__addDefaultDomainPrefix` function
Use regex for index redirect URL.
Fix: Ensure correct trace handling in test_groups_gather_equivalence
Ensure local and remote paths end with '/' for consistency
Fix calendar roundtrip issue and add check for numpy version.
Refactor: Use ceph_deploy.lsb module instead of direct import.
Add tests for escaping characters in CSV writer and reader

Fix issue with potentially too large PoEs.\n\nThis commit introduces checks to prevent disaggregation results from producing excessively large probability of exceedance (PoE) values. It raises a ValueError if a PoE exceeds a predefined minimum threshold, ensuring that the output remains within acceptable bounds.
Refine invalid response patterns for SSH autodetection
Fix: Properly destroy GoToLineDialog's C++ object on close

Simplify modify_parameters method signature
Ensure images are in RGB mode for locateAll function
Bump version to 0.7.7rc1
Bump version to 0.8.4
Allow updating array attributes from JavaScript side (# issue-number)
Add STOP_WORDS as a set

Fix: Handle None values in deployment configuration file
Increase POSITIONS_PER_XML from 500 to 1000
Refactor: Simplify directory cleanup logic
Disable automatic code reloading in shell app.

Emit clicked signal when checkbox is toggled
Add tests for deleting categories and their children.
Bump version to 2.0.4
Refactor: Add unique value tracking for method outputs.
Update jar file pattern for testing
Simplify input path handling in executor
Add version printing and project initialization functionality
Use logger.warning instead of logger.warn
Fix: Correctly handle has_ref flag in InstanceProperty
Fix: Handle potential ordering issues with ID3.get() results
Fix: Add newline at end of file
Bump nionutils dependency to >=0.3.15
Add InputBase class for input parameters definition
Fix regex encoding for delvals
Fix: Improve logging and progress bar for large directory processing.
Update libuv revision to 93877b1
Downgrade numpy dependency to 1.16.4
Add Apache 2.0 license
Fix count calculation for directories containing downloads directory
Add help subcommand functionality and tests
Convert HTML content from bytes to native string
Add environment argument to apextestsdb_upload
Fix: Handle None for feature_names in plot_feature_importances
Use OQLITEIMPORT environment variable instead of OQLITEDEVPACKAGES
Use _qsize() instead of qsize() for ExhaustableQueue representation
Add new potential wrappers: CorotatingRotation and GaussianAmplitude
Bump version to 0.13.5
Bump version to 0.9.0
Fix unicode handling in user profile tests
Fix handling of minimum/maximum in kernel symmetry calculation.
Copy handlers and handler symbols for polygon connection
Refactor: Avoid Celery for post-processing when openquake.no_distribute() is True.
Fix: Handle unsupported looping behaviour in UnstructuredJoin
Fix: Install Prism correctly using Popen and wait()
Close event loop if not supplied by user (#1234)
Improve logging for template rendering
Update headers dictionary type
Update project URL to use sr.ht.
Fix: Return empty dictionary instead of raising an error on failed Master connection
Fix type check for dimension attribute in nc class

Refactor: Improve handling of memcache response chunks
Fix dangling SocialAccount issue by checking for attribute existence before accessing provider account.
Add README.md as long description
Skip IPv6 websocket connection test due to Travis errors
Update MultiProcessValue to accept a custom process identifier function
Fix: Find matching substring in parent nodes using generator expression
Fix readpartial test to handle partial reads correctly
Refactor: Improve inline figure configuration and cython magic check
Refine PolicyInformation representation in doctests and simplify policy identifiers.

Fix invalid response handling and add keep-alive check.
Refactor: Extract `ret_config` handling from `clear_load`
Rename `gccarm` to `gcc_arm` in build paths and names
Add typing dependency
Allow specifying staging S3 bucket for CWL jobs
Enable Python subprocesses to work with expect functionality.
Add support for 0x01 as a separator character in Hive datasets
Round bar and factor sizes for improved visual consistency.

Add `calculate_markov_matrix` method for generating Markov matrix
Add 'bits' to _print_slots for StateVector
Bump version to 1.3.0 and add six dependency.
Handle send_message_chat errors and add error handling for ctl calls
Bump version to 0.13.0
Include package data in setup.py
Bump version to 1.1.dev0 and clear release date
Refactor: Update CASClientWithSAMLV1 to inherit from CASClientV2

Add register flag and skip register logic
Fix: Remove placeholder response in WsgiServerTransport
Fix handling of parentheses in multipoint WKT parsing
Add support for Philips motion sensor models SML003 and SML004
Add unit test discovery and execution functionality.
Add RemoveAdapterWithDevices method
Fix spurious dimensions in image data\n\n- Remove any dimensions from the loaded image array that have a size of 1.\n  This ensures consistent handling of potentially multi-dimensional input.

Refactor ClearlyServer initialization
Improve SPRegion behavior when learning is off.\n\nSwitch to a random SP for computation if learning mode is off and no iterations have been performed yet. This prevents potential issues with uninitialized SPs.
Initialize cached job store based on current workflow and prevent unnecessary instantiations.
Update README badge and links with new version.
Add serialization support for SpatialPooler
Fix: Handle invalid engine names in open_dataset (https://github.com/pydata/xarray/issues/2050)\n\nThis commit fixes an issue where attempting to use an invalid engine name with `open_dataset` would raise a confusing error message. The fix includes adding a check for valid engine names and raising a more informative ValueError.

Remove version banner from hosts.allow content
Use connection.capabilities instead of parsing capability response for IDLE support.
Make AbstractIterator.__aiter__ async
Update copyright information with creation year and duration.
Add file deletion functionality
Add CI support for test_led.py
Use specified port for InfluxDB server check.
Fix matplotlib plotting issue in Trajectory class (#123)
Bump version to 0.2
Remove debug print statements from send_txn method
Fix: Replace secondary IDs in references
Fix: Ensure data URI output is truncated for testing purposes.
Fix: Ensure consistent string handling in MediaFile
Fix: Add self parameter to hello method in ITestInterface

Add __repr__ methods for _TdsSession and _TdsSocket objects
Fix: Only set TCP_NODELAY if it's not a UDP socket and not a Unix socket
Add support for FAT filesystem type in parted_partition module
Use setuptools if available for setup.
Remove unused helper functions
Save document data from form
Fix: Use bit_length() instead of bitlength() for long objects
Clarify verbosity argument help text
Sort edges in graph representation for consistency
Refactor: Improve traceback parsing in BaseBackend
Fix: Decode modulecmd output

Refactor: Simplify Block API and remove get() method
Use ImproperlyConfigured instead of Exception for pygments missing error.
Add terraform dependency to test requirements
Add Python version requirements
Refactor: Use sets for exclude files in installation
Fix: Set dataset version by date if not provided
Update RiabKeywordsDialog to handle title changes and add keyword.
Add Pipfile.lock to list of lock file extensions
Add VERIFY_NULLFAIL flag for stricter checksig behavior.
Refactor: Pass arguments directly instead of returning them
Refactor: Rename dewpt parameter to dewpoint in relative_humidity_from_dewpoint function\n\nThis change aligns with standard meteorological terminology.\n\nAdditionally, added math equation for RH calculation.
Add bounds for energy_lo, temp, and chem_potential parameters.
Refactor: Retrieve single best BLASTp hit and return protein name
Update console icon to use Spyder logo
Require Python 3.4 or higher
Renamed `model_precision_recall_curve` to `model_prerec_and_roc` for better clarity.
Improve error handling for SSL certificate and key
Fix overflow connection handling on OS X
Ensure SYN_DIR is available before processing
Refactor: Separate pruning of short branches

Improve handling of image encoding in pageinfo.
Update settings for celery result expiration and token expiry duration
Fix S3 URL format for Azure blob copy operation
Bump version to 0.1.2
Fix ASGI streaming handling and content length calculation.
Use 'self.results' instead of '_results' for consistency.
Update Tika version to 1.22
Close connection in __del__ method
Refactor: Print server address based on unix_domain setting
Use new testing helpers from `tests.support.unit`
Refactor: Update BBCiPlayer plugin schema for appStoreState
Fix legend field handling in ElementPlot
Bump version to 1.3.12
Add check for task_spec before sharding dataset in DistributedSession.
Use config values for SAML key and certificate files
Add checks for connection status before sending pushes
Limit feed title length to 200 characters

Introduce train/test split for model evaluation
Use find_packages for discovering packages\n
Warn if tests are run without a virtual environment.
Improve logging when filling symbolic memory with unconstrained bytes (#<Issue number if applicable>)
Use `in` for property checks instead of deprecated methods (#123)
Allow payloadBuilder to use default format if none specified
Fix: Skip purging if no packages need removal

Add `func_name` parameter to `add_line_breakpoint` function
Fix: Reduce conformance checks in test_ogcapi_features_pygeoapi
Add support for proxies in HTTP requests
Bump version to 0.7.1
Reverse order placement for pair trade strategy
Add count and average roundoff error tracking
Fix: Raise ValueError for WholeArchive with MSVC linker
Handle missing 'To' element in WS-Discovery probe match messages.
Update psycopg2 dependency to >=2.3.1
Improve handling of file permissions for local remote
Cache property instances for improved performance.
Bump version to 1.6.1
Fix documentation build by using correct Sphinx API call
Increase timeout for AllApplicationActive wait condition to 600 seconds.
Refine MRD optimization settings for faster convergence.
Fix: Simplify string quoting in `quote` function
Use select_related instead of prefetch_related for sponsors
Add test for device property retrieval during initialization
Update description and add README as long_description
Add type hints and improve test function parameters in `test_parsimonious.py`
Fix type errors in dictregister tests
Improve performance of TokenList by removing duplicate tokens efficiently.
Improve handling of single supplementary zip file for PoA articles
Remove unnecessary docstrings and cleanup code.
Log command execution results for debugging
Fix: Handle forward slashes in package release paths\n\nThis commit fixes an issue where forward slashes in package release paths were causing errors.\nThe code now replaces forward slashes with OS path separators for compatibility.
Add v1.27.1 to LANG_RELEASE_MATRIX

Add atom id generation for exports
Fix: Convert evacuation percentage to decimal

Bump version to 0.7.0
Ensure Job instances have unique IDs
Remove hardcoded default Celery settings from FlaskLoader

Use hash of memory location instead of string representation in keccak256 calls
Update API base URL to HTTPS
Bump version to 1.4.11
Adjust call rate sleep duration
Refactor: Adjust preorder lifetime calculation based on epoch.
Reset cumulative returns after each policy update.
Fix: Handle TypedExceptionWrapper in freeze_col
Improve code linking by using full function names in place of only the object name.
Update Osram Flex RGBW device description
Bump version to 0.2.1b2 and add Python version support for IPython dependency\n\nThis commit bumps the version to 0.2.1b2. It also adds Python version-specific requirements for IPython, ensuring compatibility with both Python 2 and 3.\n
Adjust calculation of number of coarse channels based on channel width
Fix: Ensure column width is considered when creating numeric frequency table columns.
Implement `get_autosync_payee` method in LedgerPython and HLedger classes (placeholder)
Fix: Only draw waveforms if spikes exist
Refactor: Simplify share configuration in schema_external.py

Refine test description for zeep exception handling
Fix kwargs passing in state execution
Register resource views based on queryset model instead of NotImplemented
Refactor: Use context manager for m3u file creation
Refactor: Rename `__metainstance` to `_metainstance` for better clarity.
Return base unit string directly if dimension is in base units.
Clarify wording in Binary Heap docstring
Refactor: Use os.path.join for filename construction in stitch_macro
Use sets for `hosts_to_ignore` for better performance
Remove TODO comment about Click and setuptools integration
Exclude tests and tasks from package installation (#123)
Update apitools version to 0.5.10
Add TestReactorMixin to EndpointMixin
Ensure scenario ID is valid (non-negative integer)
Implement gradient for scipy.linalg.sqrtm using solve_lyapunov
Add encoding parameter to decode function
Always allow queries on secondaries in ReplicaSetConnection
Refactor: Update Process creation method for Heroku
Disable IFrame test on Windows
Bump version to 0.6.0.dev20191212
Fix ValueError handling in BoundingBox.get_slices() method.
Add test for output format inclusion across APIs
Fix: Use correct version check for strip_fragment
Bump version to 0.3.2
Update editor highlighting after color scheme change
Refactor EMS API client initialization
Avoid modifying cached data in creatTree function
Update lr scheduler to MultiStepLR
Refactor: Simplify instrument configuration handling in clirundal.
Refactor: Remove unnecessary OS upgrade configuration
Refactor: Improve git_repos subtask logic and add error handling for pip_packages

Remove PIL and datastore warnings from test output
Remove mock_raw_event function
Remove labels and help texts from AssignShiftForm
Refactor: Simplify EQI1 parameter trimming logic.
Handle calc_id retrieval from datastore even when not present in database
Fix AMI protocol version detection
Update worker pool core count reporting
Remove purge_db method as it's not implemented by any subclasses
Add matplotlib and its submodules to mock list
Add tests for param_logx_diagram and bs_param_dists, requiring fgivenx
Improve docstring of KGPotential class
Refine logging format for info messages
Add textupdate widget to AsynOutportPart
Fix: Set Pipfile hash in PipfileLock if not already set
Log connection lost and error received events
Fix: Minor wording improvement in docstring for _precache_zipimporters function.
Add team info to on_teams_members_removed method
Add error handling for starting Beam workers.
Improve documentation for KaplanMeierFitter confidence interval calculation
Ensure geohash results are included even with bucket overflow and center by limiting desired results
Handle infinite values in expression operations
Handle directory creation with error handling.
Update mapping for NIDM results attributes and fix task substitution
Refactor: Retrieve instance ID from node object instead of using a LocalClient
Attach child watcher to asyncio loop
Improve similarity check in Brain class
Add support for TigerData protocol in to_https_protocol function
Refactor: Print specific config file name during check
Ensure tests are executed only when packages are specified
Add project URL to setup.py
Use `xrange` instead of `range` for better performance.
Normalize model output before Sobol analysis
Fix typo: 'relase_resources' corrected to 'release_resources'
Fix emoji spacing for additional emojis
Extract game ID from PowerLogParser
Handle potential setuptools import error
Add libxslt dependency for FreeBSD deployment
Refactor: Remove unused address parsing functions
Fix subtitle formatting in Rinoh documents
Fix: Ensure `dj_database_url.parse` is called for all test cases.
Fix: Handle backslashes in taxon names
Refactor: Use `has_role` instead of `role_in` for clarity
Use `inspect.getfullargspec` for Python 3 and backport for Python 2
Fix: Initialize _datastore correctly in MachineCategory
