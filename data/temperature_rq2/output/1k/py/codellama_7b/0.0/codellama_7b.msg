Detach buffer and disconnect port from NodeMap in Harvester class
Update S3Manager class to use list_buckets method instead of get_list method.
Fix bug in error reporting logic to prevent sending reports from test setups and non-EC2 instances.
Improved server performance by using cProfile to profile the loop.loop() method.
Refactor Time class to use MetricUnit instead of Unit for seconds and minutes.
Update parser for Python 3 compatibility
Fix issue with removing associations from store. Fixes #1234.
Update file content for test_file.py in tests/integration/states directory
Refactor CSS escaping code to improve performance and reduce memory usage.
Reload modules and debug settings
Update copyright information and version number in conf.py file
Update paths for new repository and timespan for benchmarks
Added option to disable automatic cross-seeding and added option to set an explicit label for cross-seeding (changes info hash).
Added support for MAGICC7 and above by allowing users to control the level of logging with the 'debug' argument. Also added a warning if the user attempts to use 'debug' with MAGICC6.
Added force_renorm option to renormalize spectrum in case of partial overlap with passband.
Refactoring of get_file_chunk_consumer function to improve performance and reduce memory usage. Removed unnecessary code related to file descriptors and added a new parameter for the encoding and decode errors.
Added support for minimum length of BPDU packets
Enable reporting for all input pins on a port
Update SortedHostCollection class to handle new host object type
Added support for legislative input in Column class.
Added logging for errors when getting vault information from Braintree.
Update OBO parser to handle new format version and data release number in the header.
Added support for customizing color values in Powerline Bash prompt.
Update address validation and decoding methods to use classmethods instead of staticmethods
Update doc tree metadata parsing logic to handle invalid YAML blocks
Fix test for extract_page_links and extract_content functions in toolbox_html_tools module.
Update send_file function to support caching and etag generation
Fix heartbeat exception handling in protocol.py
Update method.py to apply response formatters to the output of the request made using the calling function returned by the module attribute 'retrieve_caller_fn'.
Added probe service and constants for monitoring message reception, emission, and content
Added support for grouping metafeatures by type (landmarking vs. all) and added a new method `list_metafeatures` that takes an optional `group` parameter.
Update regular expression for parsing CRTF files to include optional version number in the beginning of the file.
Refactored ControlConnection to use a new connection object if available.
Fix issue with mkdir_p function creating duplicate directories
Update setup.py to include release and upload_docs in sphinx requirements
Update version number to 1.0.2
Added support for Darwin (MacOS) as a valid platform in the platforms() function.
Reconnecting client after registration to server
Update version number to 4.0.2 in semantic_release/__init__.py
Bump version to 0.10.1
Added tests for multi-dimensional indexing and slicing of arrays
Fix: Remove unused dependency on `sos.converter`
Added support for default broker and client configurations in setup.py
Added flake8 ignore comment to Django models file
Update setup.py for OSX compatibility
Added support for specifying a custom shell command for new windows in `Session.new_window`
Update wheel dependency to 0.30
Update matplotlib backend to Agg for test visualization.
Fix parsing of TTL files with multiple formats
Added support for named profiles in AWS EC2 connection
Added a new tag for capturing template output into a variable; useful for including complex content into a blocktrans
Update PyMemcache configuration for testing
Update version information for EUL Fedora
Update heatmap options for better performance and visualization
Added test cases for code blocks and syntax boundaries in CMark parser
Added new voice data for Polly synthesis
Added support for Read the Docs environment variable to detect if we are on Read the Docs and insert the project's root directory into sys.path.
Added support for saving datasets as NinjoTIFF files using Dask's delayed function. This allows for parallel processing of large datasets and improved performance.
Update Git formula to fetch and merge repository instead of cloning it.
Update thumbnail size in image.py
Fix unexpected engine output warning
Improve performance of `DataFrame.any()` and `Series.any()` by adding a new parameter `bool_only` to include only boolean columns.
Update Firewall class to use new Action.to_rest method with DP parameter
Update Kubernetes tests pipeline to use newest Dagster test images
Fix issue with .get_object() returning all domains instead of requested domain
Refactor utils.py to use OrderedDict instead of six.moves.urllib.parse for compatibility with Python 3.7+
Added support for setting the root URL of the JSHost instance
Refactor open() method in Path class to accept keyword arguments like io.open()
Update spec_cases.py
Fix issue with homely._utils module and HOME environment variable
Fix issue with OpenSSL connection handshake
Handle property definitions in _impl phase
Add support for ellipse regions in CRTF file format
Added assertion for empty response content in test_notrequests.py
Fix VCF file header for VarScan output
Update ConsumerThread class to pass config in constructor if available
Refactor boolean_and_simplifier to use _flattening_filter and check if we are left with one argument again.
Refactor session class to use type hints and remove unnecessary imports
Fix parsing of temporal attributes with dateutil library
Update build_type function to handle gcloud-python source path correctly
Refactoring of tile directory format to improve performance
Added tests for checking allowed requests in OWSSecurity class
Update ResNet backbone with FPN and freeze specified number of layers
Added support for custom holidays in `Calendar` class
Added support for reading empty arrays of different dtypes in Parquet files using pyarrow. This fixes #41241.
Added support for checking permissions on actor objects
Added function get_token_features to extract features from tokens for gender matching.
Fixing issue with copying whole columns and rows in ArrayEditor widget.
Fix bug in fusing projections
Added support for caching and logging in API Gateway stages
Refactor orderby clause in SqlaTable model to use direction parameter instead of hardcoded ascending and descending values.
Added support for calling methods on lenses with a shortcut syntax
Update source file suffix and parser in conf.py
Update manager.py to wait for droplets to be created before setting job state to 'droplets-active'.
Added error handling for OSError in real_size function to handle non-existent files and added try-except block to get size of file.
Update test cases for Eyring equation with linearly ramped temperature
Added support for specifying sstable size in MB using `__compaction_sstable_size_in_mb__` attribute on model class. Fixed issue with `min_sstable_size` option not working.
Increased benchmark iterations from 1000 to 100000 for more accurate performance measurement.
Fix issue with integer and binary variables in GLPK solver
Normalized arXiv category strings using new mapping
Update handle version number on revision commit
Added support for tracking the highest parent of a parameterized object's parameters. This is necessary for proper handling of nested parameterized objects.
Update validator manager startup script with delay-start option and redirect stdout and stderr to files
Update HPC config for pename and memtype attributes
Update Potential.py to support plotting of non-axisymmetric potentials and effective potentials.
Fix issue with managing package repositories using salt.states.pkgrepo module.
Update URL patterns for server views
Fix parsing of HTML tables with lxml. Remove unnecessary code and exception handling.
Added support for a whitelist of statements to process in _process_Stmt() method.
Update version number to 0.3.3 in setup.py
Added support for MacOS source display in Object Inspector plugin
Refactor upscale function to use pdfrw instead of pypdf3 for improved performance and stability.
Added project URL and classifiers to setup.py
Bump version to 1.8.6
Resolve SRPM for RPM ID: <rpm_id>
Update version number to 0.3.1
Update file.py state module to use new 'file.append' function
Update version number and JS version number for release 0.5.0-beta.1
Improve performance of search_unit_in_bibwords by using run_sql instead of run.
Update HolodeckEnvironment class to handle tick limit and display environment variable
Improve documentation for `approx` class in `_pytest/python.py`. Add information about the differences between different algorithms used for comparing floating-point numbers.
Update GCE cloud provider to use sudo_password for non-root users and display SSH output by default.
Added a new method 'actionStop_triggered' to the Container class and called it from the execute() method of the Algorithm class.
Added support for local mode in ApiConfig class
Update country flag image URL based on selected country code
Refactor primary_avatar_object template tag to use a constant key instead of a variable.
Added a new method 'foo' to the Motor class
Fix detach method for sources that are already detached
Update Witness_update class to handle empty block signing key correctly
Update PEP517 hook caller to handle Python 2 environment variables correctly
Added __hash__ method to ContextOptions class to ensure compatibility with datastore_rpc.Configuration.
Fix test for file cache expiration
Added support for disaggregation of electricity demand using the etrago model.
Update git config to ignore current branch during release minor version bump
Added support for saved installation defaults during upgrade.
Added support for encoding ASN1_force objects using the enc method.
Refactoring of flask_login.py to use new init_app method
Update dependencies for Python 3 compatibility
Update KotlincPluginIdField to remove required=True
Added support for parsing Thermo raw files using MZMLIterator class.
Update DateTime fields to use timezone-aware format
Update JSON schema for component parameters
Update state module loader for remote compilation via ssh
Update couch_to_ntuple.py to write ntuples instead of filling them
Issue 88 - Resolved by adding set_session() and self.sess for KerasRunner class.
Fix duplicate waypoints in WPModule module
Update invoice date format to ISO8601
Add MixtureSameFamily to PYTREE_BLOCKLIST in jax_transformation_test.py
Added support for using requests library instead of urllib2 for fetching data from http://www.schneierfacts.com/
Added support for Google, MSN, Yahoo, and Ask Jeeves bots in UserAgentParser class.
Update verbose optimization class to use ipython notebook as a default value for the `verbose` parameter.
Fix swapped current electrodes in EIT40 importer
Update ReceiptPDFView class to use new ReceiptPDFView class instead of ReceiptPDFDisplayView class.
Added check for name, email and username in enrich function to handle cases where identity does not contain any of these fields.
Added special login handler for Arista devices to handle 'Last login' message.
Update CifWriter to use significant figures for formatting floats instead of float precision.
Update version number to 2.0.0
Fix status.pid function on Windows platforms
Refactor: Improve performance of pyprf_opt_brute by reducing the number of unique models and improving the accuracy of polar angle calculation.
Added support for customizing the environment variables of tools. This allows users to specify additional environment variables that are required by their tool.
Added wafer.compare to INSTALLED_APPS in settings.py to fix a bug where Django wasn't finding the overridden templates.
Update prep_bootstrap method to return a tuple of two values
Update introspection query to include description and deprecated fields
Update logging service to use new Elasticsearch query API
Update setup.py with encoding support for README.rst file
Update photo url property to use custom image URL prefix if set.
Added support for indexing colors by position
Update file ID generation for shared files in AWSJobStore
Fix issue with exception during record halting playback. Add timeout parameter to get_next_data_elements_to_finish() method to prevent infinite loop.
Fix TypeError and ValueError in int_to_roman function
Refactor Executor initialization and add timeout support for execute_action() method in Context class.
Update minimum setuptools version requirement to 3.7.0
Improve parsing of format specifiers for integers in FormatString class.
Bumped version number to 0.8.1
Refactor daemonize function to allow for custom log directory
Refactor lattice comparison to improve performance and fix a bug.
Update build_docs.py to check for literalinclude directives in example DAGs
Update templatetags for Django 1.9 compatibility
Update __init__.py file for positioning module
Added support for disabling timeouts during package installation and removal to improve safety.
Update BioGrid version number in dataset metadata
Improve clock synchronization between client and server by introducing a new parameter '_CLOCK_DIFF_MAX' that controls the maximum acceptable difference between the client and server clocks. This change improves the accuracy of the polling mechanism and reduces the likelihood of errors caused by differences in clock time.
Added print statements for debugging purposes in BaseAttRelTestCase class.
Update SlurmCluster class to support customizing number of CPUs per experiment
Update subprocess module imports and add timeout for command execution
Improve performance of test case by increasing number of entities and beats.
Added validation for JSON field to ensure it is a valid JSON string
Added support for creating directories when using touch
Fix loop condition and body linking in FunctionSolc class
Fix incorrect classpath element in test_zinc_utils.py
Update extras_require in setup.py for dev and test dependencies
Update README file path in setup.py
Fix race condition in MuxConnection.connect() by adding a lock around the socket connection and increasing the retry count for connect().
Update utils.py to use collections.abc instead of collections.
Update version number to 0.0.6
Update include file syntax to support newlines in filenames
Update version number to 0.14.1
Refactor testmagics.py to use new IPython API and remove unnecessary dependencies.
Update SingularityParser to handle environment variables with '=' correctly
Update Consumer class to handle invalid auto_offset_reset values
Fix issue where PVD unused fields are not zeroed out correctly
Added support for listing all available plugins and presets in PluginHub class.
Added support for Luciddream build type in buildbot.py
Update version number to 0.1.17
Update version number to 1.1.dev in setup.py
Fix log file handling in engine.py to ensure that it is writeable before running the calculation.
Refactor setup CLI command to use new Dusty API for config management and VM size recommendation. Remove unnecessary tearDown method.
Refactor SRT file parsing and formatting logic to improve performance and reduce memory usage.
Added support for PyQt4 and PySide API
Update version number to 0.0.7
Update status callback function to use new RLBot API
Update CatTransformer test case to check for unique keys in probability_map
Fix: Remove unnecessary code and simplify the logic for determining whether a request is blocked or not.
Update bootstrapping module to use boltons.funcutils.wraps instead of functools.wraps
Update dependencies for ckmutil and wcxf to latest versions (#1234)
Update champion spell data to use new cost type field
Update option help text for --absolute-time in globus endpoint is-activated command
Added support for retrieving properties of a Subversion repository using the `svn.local` module.
Fix incorrect usage of get_ad_metadata() in get_ad_item()
Fix timeout handling in session.post() by using time.monotonic() instead of time.time()
Fix negative sigma initialization in NormalMessage class
Added support for plotting qubit layouts using matplotlib.
Update Sphinx configuration file to allow for proper documentation generation
Fix integer overflow issue in merge_asof() by casting values to int64 before comparison
Update logging level for cluster definition from file and service url
Update flightgear integration to use correct heading calculation for GPS data.
Update VTK-MeshIO plugin to use new API for converting between VTK and MeshIO data types.
Assigning generic geometry to entire domain if no pores and throats are given
Fix scaling issue in CFScene class by adjusting offset and scale values.
Update copyright notice and add 2019 as a valid year
Update version number in __init__.py file
Refactor chunk encoding to improve performance and reduce memory usage.
Remove unnecessary Python3 compatibility code and update package structure
Fix duplicate sample name issue in verifybamid module
Added support for multiple languages in search index factory.
Removed unused code and updated controller destruction logic.
Update tax rate for invoice items with no tax rate set
Added support for reading images from file-like objects in `dispatch.modules.content.models.Image`.
Added security warning for running testapp in production
Update salt-cloud command to use new configuration file format
Added withdraw method and updated balance method to use private query for Bittrex interface.
Improve performance of exec_command by reducing the number of channel events handled.
Add support for EdX opaque keys
Update CSRF check to use HTTP 403 error instead of 400
Update celery beat schedule for intermittents commenter task
Fixing issues with weather command and adding error handling for invalid locations.
Added tests for BoundingBox.get_overlap_slices() and fixed a bug in test_bounding_box_slices().
Fix download feature data and metadata at well API endpoint
Refactor FileLock class to improve performance and reduce resource usage.
Added error handling for missing .frigg.yml file and added a runtime error if no tasks are found in the build configuration.
Fix issue with zipped AHI HSD files not being properly cleaned up
Unfollow non-followers
Update get_path method for model
Added support for filter parameters in CouchDB river configuration
Update version number to 0.2.1 in precise_runner/__init__.py
Update version number to 2.1.3
Fix error handling for SQL queries
Refactoring encode_task_experimental function to remove unnecessary code and improve readability.
Update version number to 0.1.0a2
Initialize CA with new key and generate OCSP keys and cache CRLs
Execute state machine until specific state. Asynchronous task.
Fix test_fork_exec to use sleep 0 instead of sleep 1
Update vertical effective stress calculation in SoilProfile class to use numpy where function for better performance and accuracy.
Added support for custom timeout and callbacks in entrypoint_waiter function
Refactoring of Callable class to support keyword arguments and positional arguments. Added memoization for improved performance.
Update MySQL connection password handling to allow for empty passwords.
Added support for creating containers with CPU real-time options in API version 1.24 and added tests for it.
Added validation for YubiKey setup view
Update Markov library import statement to use new class name
Added support for storing and retrieving results asynchronously using the `AsyncResult` class.
Optimize bulk insertion of hazard curve data for classical hazard calculation
Added support for using a custom queryset in NestedHyperlinkedRelatedField and NestedHyperlinkedIdentityField. Also added a new method `use_pk_only_optimization` to disable the optimization.
Update V2_TO_V1_MAP constant to include new supported OS classifier tags
Added test for isinstance() with Integral types in Python 2
Update impact layer file path in RiabDock class
Added support for disabling privileged exec mode using a customizable command (#1234).
Added support for daemon mode in Stream class. The `daemon` parameter can now be set to True or False to control whether the thread is a daemon or not.
Update legend documentation with custom default handler for Line2D objects
Fix incorrect dependency detection for relative imports in test_relative_imports.py
Update asset copy for new singlehtml builder
Update Python script for Bittrex orderbook API using asyncio library
Added __unicode__ method to KeyPublicationStatus model to improve debugging and logging.
Update version number in setup.py
Fixup for ROOT's error and warning messages to improve consistency of errors/warnings
Initiate server for master and slave gateways on different ports
Fix invalid argument handling in command plugin
Fix issue with user authentication cookie not being decoded correctly.
Added support for multiple commits during autosynth generation
Refactoring of the publish method to use asyncio.get_event_loop() instead of creating a new loop.
Update GIT_ATTRIBUTES for BIDS datasets to use SHA256E instead of MD5E and add support for larger files (>1MB) in dataset_description.json, LICENSE, and README.
Improved charting performance by reducing the number of draw calls and optimizing the bar rendering algorithm.
Added new packages and package data for futurize.py and libfuturize.py, updated README.rst and added useful_links.txt
Update version number to 1.3.0
Update version and git commit information for PyAV
Refactor duration calculation in segment and segmentlist classes. Deprecate old duration method.
Added option to disable SSL certificate verification for API calls
Added test case for receiving a random control frame and checking that it is properly handled by the FrameProtocol.
Update output directory URL for opening on file explorer
Update course settings: enable student group choice and classroom usage based on form input.
Update pandoc conversion for ipynb files to use utf-8 encoding
Added tests for creating and using a TorConfig object from a TorControlProtocol implementation.
Update next period date end calculation method
Added support for empty results by adding a status column if all columns are hidden.
Fix a bug in the handling of directory templates. The code was modified to use `yield from` instead of `for` loop to generate the list of files and directories.
Update socket directory for master-side communication
Improve handling of missing grid mapping variable in xarray.parse_cf() method.
Update REST API for Salt using CherryPy framework to support various output formats and content types
Update regex pattern for parsing version numbers in zypper.py
Added support for using PlaceholderField and ContentItemRelation in ContentsEntryMixin
Update arbiter daemon to use new method for getting all states
Fix issue with getpixel() function when using negative offsets. Fixes #1234.
Fix issue with printing properties in sudsobject.py
Fix SNMP timeout error handling in test_netsnmptestenv.py
Update `install.py` to use `runNpmInstall()` with optional arguments
Refactored `run` function to improve performance by reducing the number of times the `result` variable is accessed.
Update version number to 3.1 in __init__.py file
Update particle positions and radii in global state
Fix parsing of expressions in vaex-core/vaex/expresso.py
Added command line interface for running map editor and loading default database file.
Added __unicode__ method to CssBackground model to display image URL in admin interface
Added support for 3D WCS transformations
Update setup.py with new dependencies and remove include_package_data flag
Update HealthEventFilter to support filtering by upcoming events
Update supported versions for Application facade in client connection
Refactor SegmentationPurityCoverageFMeasure to use new base class and improve performance.
Improve attach stream and cancel timeout handling
Added support for custom titled/commented import sections in isort
Added support for ordered dictionaries in SphinxMapperBase class
Fix external table name generation to use f-strings instead of format() function
Update license information in setup.py file
Update CLI script directory for tests
Added support for Ceph and Calamari monitoring plugins. Added copy specs for Ceph configuration files and logs. Added commands for Ceph status, health detail, OSD tree, OSD stat, MON stat, and MON dump. Removed Ceph keyring files from the forbidden path.
Fix DataProc job status check to handle API retries
Refactor find_symbol method to use find_market method instead of hardcoded symbol search
Refactor HMM initialization and add support for sub-HMMs
Update test data for magicc module to include new year values in metadata and timeseries.
Added support for large files in ISO format and updated tests accordingly.
Update dependencies for snorkel-metal and spacy
Update pip dependency to version 10.0.0
Update pytest fixtures to support function, class, and module scopes
Added support for more complex numbers in pt_BR locale
Add support for iterative imputer in FancyImpute library. This includes enabling the experimental feature in scikit-learn and adding the IterativeImputer class to the __all__ list.
Update known hosts file for SSH connections
Added support for RHEL and Fedora distributions in SUPPORTED_DISTROS constant.
Update Bayesian optimization model selection
Update minimum magnitude for sources in readinput.py
Fix negative temporal parameters in audio region splitting
Update long description of package in setup.py
Update version number in Flask-RESTPlus API object
Update test runner to ignore hgroup elements and follow W3C spec
Refactor native model parsing to use a more efficient approach. Remove unused code and simplify the codebase.
Added support for default image configuration in CompositeBase class
Update Twitter API endpoint and remove unnecessary header authentication parameter
Update waterfall status resource to include footer
Added support for remote user authentication in login plugin
Refactoring of ZipExtractor class to support extraction to a specified directory
Update impact functions list in test registry
Update HTTP request logging to hide sensitive data
Update __init__.py to include new imports and remove unnecessary ones
Fix test_xatop.py for SQLite compiled without SQLITE_ENABLE_UPDATE_DELETE_LIMIT
Refactor ArticleAdmin class to use newman.NewmanModelAdmin and remove ordering field
Added support for loading models saved in Flax format.
Update makeIdentifiers function to use outputDir argument instead of hardcoded output directory path.
Fix task execution and deletion in TaskQueue class
Update Blobstore Tests to use new stub API
Update setup.py to use setuptools instead of distutils for Python 3 compatibility
Update version information for swauth module
Update memory format and divisor for MiB display
[CI SKIP] Update setup.py for Python 2.7 compatibility
Update websocket routes for reverse API and event handling
Refine deploy task and perform necessary actions before generating context.
Update setup.py with new version and long_description function
Refactor element_methods module to include advanced XML manipulation methods based on lxml.
Fix bug in circuit building logic by using correct hex representation of router IDs.
Added support for XLIFF file format in POEditor API client
Update functionTagProcessor class to use superclass constructor
Update permissions for Bika Generator setuphandlers.py
Added support for running Brian simulations with PyPET
Update frame plotting function to use subplots instead of figure and close. Fixes #123.
Update input embeddings in BartModel to use shared embedding layer.
Update coincidence workflow to use new thinca output files
Update Project class to support building from tags
Update license in setup.py from Apache-2 to Apache-2.0
Update test_assertion.py to use new truncate explanation format
Update IP address detection logic for non-Windows platforms
Update betamax and mock dependencies for compatibility with Python 3.9
Added support for two-factor authentication
Added functionality for creating and deleting addresses in a wallet using the `neo-python` command line tool.
Refactor test_elapsedtimer.py to use pytest.raises() for timeout error checking
Update normalization parameter in correlogram view model and update visual accordingly.
Update Python language configuration for MacOS to restrict number of interpreter versions tested.
Improved performance of EOS mix methods by reducing memory usage and improving computational efficiency.
Added support for empty JSON objects in simulation initialization
Update Kubernetes check to retrieve container limits and requests metrics
Added support for parsing years and months in addition to days, hours, minutes, and seconds.
Update esptool.py to support flashing images larger than 4MB
Update logging format to use percent style multiline with auto indent
Refactor CRUDResponse class to remove hidden attributes from flatten method
Update Pipe class to include LambdaComponent support
Update version number to 0.21.0 in statik/__init__.py
Refactor readinput.py to use parallel_split_filter and shutdown Starmap
Update lineage colours for Pangolin module in MultiQC
Fix timezone offset calculation in FixedOffsetTimezone class. Remove unnecessary lines from the code.
Added support for temporary directories on Windows using `tempfile` module.
Upgrade login account from version 1 to 2 to support non-ASCII passwords.
Added support for specifying endpoint bindings and locations in metadata configuration files using a tuple of (location, binding) or a single string representing the location. Fixed an issue where the index was not being properly set when using a tuple.
Update client.py to handle HTTPS endpoint errors
Added support for deploying Mantis applications with a specific name
Added check for entity ID in PlayerManager.add_player() to prevent duplicate players from being added.
Update connection string for Patroni controller
Update remote repository working tree for StarCluster plugin
Improve performance of align_and_parse function by reading in whole file and returning 0 if no data or no clusters found.
Added support for Linux platform in _GetNativeEolStyle function
Update certificate verification logic in Salt API's CherryPy REST interface
Update version number to 1.1.4
Update OMIM source to include new type for phenotypes and predicated phenotypes
Fix bug in NegativeBinomial distribution's log_prob method when using auto-vectorization.
Added support for waiting for ready state complete on page loads when setting form values.
Update decision rules states class to handle 1D input points
Refactor ping transmission command generation. Remove unnecessary code and simplify the command list generation.
Fix issue with applying configuration on IOS devices by setting base prompt before checking for errors.
Update test data URL for Python 3 compatibility
Update TaggableManager import path for taggit_selectize and taggit_autosuggest
Fix parsing of single-item grids in hszinc.parse() method
Added support for setting fields from positional arguments and options
Update pre_grlsd_loop kernel plugin for XSEDE Stampede2 environment. Add MKL library path to LD_LIBRARY_PATH.
Added support for new target types and updated existing ones in MbedBoard class.
Update buildkite push and tag images logic for merge builds
Update pre-processing pipeline for XLM-R model using sentencepiece model and torchtext's transforms.
Update test_gerrit.py to include new changes from Gerrit
Apply a filter to this `Spectrum` in zero-pole-gain format.
Added integration tests for pyads module using unittest framework and added a wait time before opening AMS port to avoid error on Windows machines.
Update NER tagging function to return list of tuples for output_type=list
Added support for OAuth2 authentication using django-oauth-toolkit
Added save() method for request object in test_homepage_requests_filled() method of FromHome class.
Update version number and download URL in setup.py file
Added assert statement to ensure that rundir is configured before using it in get_pidpath function.
Update M4 lexer name to 'M4' for consistency with other Pygount lexers
Refactor JustinTV plugin to extract channel name from URL
Added support for parsing date fields in API responses
Added support for grouping by multiple columns using a tuple or list of column indices.
Added support for FTP target connection and file transfer using the ftplib library.
Update error handling for fastavro.schema module
Added support for unreleased items in changelog generation
Update version number to 0.8.1
Added concatenate method for StructuredGridFilters class to allow concatenation of structured grids.
Update Trac API endpoint URL with authentication credentials
Added support for closing Dask client and cluster resources when the resource is no longer needed.
Added support for Python 3.6 and 3.6 notebook environments in PykeChain.
Fix import of stl module for marching cubes algorithm
Update config model with new config file
Add psycopg2 as a dependency for cauldron
Update template expansion function to include timezone information
Fix issue with NTLM authentication on Python 3 by encoding URL and body as bytes before passing them to urllib2.Request.
Update signal for build jobs to stop experiment when all jobs are finished.
Added support for writing binary data to stdout in SafeOutputWrapper
Added support for Python 2.7 by using backport of `lru_cache` from `functools32` library.
Update version number to 1.14.7 in setup.py
Added support for sending notifications to multiple users
Refactor omit method to handle dictionaries correctly
Update diff.py to include new classes and modules in Scala codebase
Update RegtestDaemonService to use generate instead of setgenerate
Update harvest command to use DEBUG setting from Django settings
Update astroid package version to 2.3.0.dev0
Added a new function `randomizer_bin_und` to generate random undirected graphs with a given number of edges and a probability parameter `alpha`. The function takes two arguments: `R`, which is a list of all possible edge indices, and `alpha`, which is the probability of an edge being present in the graph.
Update version number in setup.py
Refactor test_aardvark.py to use unittest instead of nose and mock.
Update readme maker to support custom line break count for chapter headers
Update path library dependency to use newest version
Added HazardMapDBReader class to read hazard map data from the database and serialize it into an XML file.
Fix issue with sending and receiving data over a unix domain socket using the txdbus library.
Refactor request body processing to use urlencoded_processor instead of raw_body
Refactor PandasQueryCompilerView to use PandasQueryCompiler instead of reimplementing it.
Update renku version in git config
Added support for querying bugs fixed in a specific release
Fix issue with reading CSV files by removing unnecessary print statement and adding logging info
Update display layers inspector section to handle various data groupings
Added customized_warning function to display warnings with additional parameters file and line
Refactoring of readinput.py to support parallel processing of composite source models.
Fix import of unittest.mock and add try-except block to handle ImportError
Update iface command to require datacenter and make ip version optional.
Update downloader from ChromeDownloader to WgetDownloader
Added test cases for special values of DeVosSolarcell calculator methods
Refactor debug decorator to use functools.wraps and print output to stream instead of stdout.
Added support for one-step synthesis of monomers in PySB assembler.
Update HTMLMinPlugin to support minifying files with non-UTF-8 encoding
Bumped version number to 0.2.2
Fix socket shutdown and close for MQTT client disconnection
Added support for displaying email previews in the admin interface using Django's `mark_safe` function.
Added support for passing positional and keyword arguments to handler constructors. This allows developers to customize the behavior of their handlers by providing different arguments when they are instantiated.
Update BaseModelResource schema to include object_class attribute
Load Saltfile from user-provided path
Added support for adding and concatenating forces with lists of forces
Update documentation generator to use new formatting utilities from Rest Framework
Fix bug in hessenberg decomposition via Householder transformations (#3364). Implement algorithm TODO in [GolV13]. Add support for Givens rotation.
Fix issue with Python 3.x compatibility for reading text files as binary data.
Fix issue with .get_object() returning all domains instead of requested domain
Refactoring of high-level function lookup in convert_type_of_high_level_call() to improve performance and reduce code complexity.
Fix SSLWantReadError handling in HttpUA class
Added support for setting Windows taskbar icon
Added explicit encoding for Windows compatibility in utils.py
Remove unnecessary code for skipping columns with Foreign Keys in ModelConverter class.
Update ReaderEventNotification encoding and decoding functions
Update lint.py to support new feature of generating reports for individual files
Fix upload order in CachingFileStore.update() to ensure correctness and efficiency.
Refactored cancel_listener method to use a more efficient algorithm and added support for filtering by view factory.
Update StatementAdmin model to include created_at field in list_display and list_filter
Fix issue with aborting connection when transport has no abortConnection method
Added new function `new_random_token()` to generate a random token for user authentication.
Added support for calculating the average power spectrogram of a TimeSeries using the Welch method with cross-spectral analysis.
Added support for visualization of suspects using the `viz` module.
Update dependencies for itella module
Update D4S2Api and D4S2Item classes to handle errors with more detailed information.
Added support for creating pending tasks and running them in parallel
Added support for randomized minion IDs to improve test stability and reduce false negatives.
Added Sentry404CatchMiddleware to catch 404 errors and send them to Sentry
Update test_get_running_exec_lldp method to use exec_command from lldp module instead of calling lldpctl directly.
Refactoring of perturbations_higher_order.py to improve performance and reduce memory usage.
Added support for displaying battery alerts using desktop notifications.
Update public key file with latest changes from private key
Added get_items() method to Cart class to retrieve a list of items in the cart.
Update repo class to handle git version output parsing correctly
Improve RDAP bulk lookup performance by reducing the number of queries performed and improving query depth.
Update get_statement_jsons_from_hashes function to use a subquery for filtering by mk_hashes instead of using an IN clause.
Fix issue with missing model in comment post view
Update image name in tag_and_push_image method
Update MonitoringCollector class to handle agent shutdown and uninstall properly
Update AioHttpInterface class to set default Content-Type header for delete method
Update notebook URL parsing to handle newlines in output
Update analysis.py to include new healthcare access metrics and fix formatting issues.
Added new test modules for TensorFlow and Caffe
Improve completer performance by using a more efficient regex pattern for matching triple-quoted strings and comments.
Added support for Python 3 unicode literals in entity_test.py
Update figure size in run_data_clean_up.py
Update test_restapi.py to use new sha3 function from raiden.utils
Fix invalid argument handling in SimpleSQLite init
Update AESFuncs class to use load[key] instead of load['key'] in _return method.
Update LSF batch system memory parsing to use correct units (MB -> GB)
Fix serial number formatting for CRL creation
Update RandomScale augmentation to use collections.abc instead of collections for compatibility with Python 3.9 and above.
Add support for Python 3.7 and 3.8 by adding compatibility with builtins.ExceptionGroup
Added test case for resolving inexistent relationship and added a new test case for improperly inherited relationships.
Update total consumption method to handle legacy protocol issue
Update logging configuration for requests library
Update ArcGIS REST API client version to 2.0.120
Update deprecated_call function signature to use Union type for expected_warning parameter
Added support for variables not present in the structured solver's adjacency dictionary.
Added support for BeautifulSoup 4 in setup.py
Added support for handling HTTP requests with a null body
Update evaluation metric for integer target labels
Update Lambda function runtime and role validation in policy.py
Added support for Django 3.0's new `input` function in the `tenant_schemas` package.
Fix issue with ARMET connector's resource class not properly binding the Flask-SQLAlchemy session.
Improve HTML repr for DataFrame in IPython notebooks and QtConsole
Added support for themes in ThemedTk class
Added validation for API key in Transport class initialization.
Added support for serving static files using pkg_resources
Update serializer type for write-only fields in documentation generation
Clean loaded libraries and refresh libraries after refreshing selected state machine
Update multiqc report generation to use new directory structure
Transfer build caches from one ODB to another
Update driver for virtualchain library and add support for ECPrivateKey from keylib.
Remove compatibility wrapper for old PyWin32 errors
Update MultiDict class to include medians and variances calculation methods
Update test_sphere_sph.py to test for schmidt seminormality instead of normality
Refine SimTypeFixedSizeArray._refine() method to handle size of elements correctly.
Update subprocess call to include stderr output
Update RIAClipper GUI with new imports and dependencies
Improve logging functionality by adding support for colored output. Add a new parameter 'force_color' to force color output even if the terminal does not support it.
Fix test for Python 2.6 and virtualenv environments
Refactoring of pymomo.scripts.momo.py to improve performance and reduce memory usage.
Refactor softmax and cross-entropy functions to use pycuda's cumath module for numerical stability and performance.
Fix issue with incorrect handling of edge cases in deci2sexa function
Update WDItemEngine class to use json.loads instead of json.load for parsing JSON data from Wikidata API response.
Update temperature in QuantyDockWidget and update nPsisAuto and nPsis accordingly
Added support for SSL connections in interactive_server() function of mockupdb module.
Fix bug in task management for DisaggHazardCalculator
Update HTMLRenderer class to handle list tokens correctly
Update constellation module and instrument lists
Fix connection pool size in PostgresStore.connect() method.
Added test cases for clearing surroundings in TestToolboxClsGridLife class
Update maximum name length for Travis setup steps
Improve restart app process by reducing wait time for reset subprocess
Update version number to 0.4.12
Fix parsing of HTTP headers with leading whitespace in Python 3.x
Refactor find_link_by_* methods to use XPath instead of Selenium's built-in methods
Added support for iterable data types with an iterator interface in DictInterface. Ensures that the interface does not consume data of other types with an iterator interface.
Bump version number to 0.1.4
Update OTT flag sets and sources
Added support for host.docker.internal on Linux to emulate Mac + Windows behavior for communication with other mock AWS services running on localhost.
Update recipe generation for new process descriptions and added tasks
Update win_service module to support non-Windows systems
Update documentation configuration file to include new sidebar options
Added support for wildcard matching in oEmbed providers
Update ssdeep library to handle internal errors
Improve performance of reachable_frontier_from by using a more efficient data structure and reducing the number of operations considered.
Added dependencies for matplotlib and wxPython to MAVProxy setup.py file.
Fix runtime error when closing writer with empty set
Update required packages for pysk8 setup.py file to use pyserial instead of serial.
Update TIMEZONE_IDS fixture in conftest.py to reflect changes in TIMEZONES list
Refactor event classes to simplify code and reduce complexity. Remove unused event_classes dictionary and add new_event function to wrap raw gRPC events in friendlier containing classes.
Disable console flow and add warning for OAuth out-of-band (OOB) deprecation
Added class method for setting up and tearing down the API server process.
Added checkpointing functionality to save the best model based on validation accuracy and remove old models when the limit is reached.
Update get_user_data function to use frozenset for storage parameter
Added support for custom queryset classes in MultilingualQuerySet. Closes #123.
Added support for checking out different branches and added tests for it.
Added support for autonaming episodes with non-integer episode numbers. (#1234)
Added support for scipy interpolation in LinearInterpolation task.
Update assess_deploy_webscale.py to report metrics for MongoDB and Juju versions
Fix bug in isChanged() method of modify.py file
Added support for DocBookBodySubNode
Fix incorrect file name in test_cloudvolume.py
Added support for sending email attachments using Django's database backend.
Added menu item for loading map and horizon in MAVProxy console.
Added support for sleeping before executing a command in a pane or session. Added tests to ensure that this functionality works as expected.
Update test_api.py to include new resource list assertion
Update setup.py to include deprecated package name
Added x and y labels to trigger timeseries plot
Update database connection string for scicrunch registry sync
Update serializer to handle missing fill value in NDArrayField
Fix issue with GT not following PAGE standard by removing PrintSpace from border calculation.
Refactored HouseStatusMonitorEnable and HouseStatusMonitorDisable classes to use a common base class
Update URL parsing library to use `urllib.parse` instead of `urlparse` for Python 3 compatibility.
Fix connection error handling in Docker plugin
Refactor notifiers/providers/__init__.py to remove hipchat provider and simplify imports.
Fix inter-process mutex for concurrent device open in ICS NeoVi can backend.
Fix issue with nested YAML values in test responses
Fix issue with PDF generation by removing unnecessary DEBUG variable and updating error handling
Added focus_set() method to canvas object in human_action.py to allow user to set focus on canvas after sending command.
Added support for using Numpy arrays in JavaObjectUnmarshaller. Importing numpy as np and suppressing any ImportErrors that may occur.
Improve symbol indexing performance by sorting package aliases
Update webhooks model to use a dictionary instead of a set for event types and signals.
Update protein sequence calculation method to remove trailing asterisk
Improve performance of scan_dir function by using multiprocessing Pool with CPU count as the number of processes.
Added tooltip background color #ffffe0 for better visibility.
Fix issue with converting pixel data to wx.Bitmap by returning a string representation of the bytearray instead of the actual bytearray object.
Update cut_release.py to handle changes in Maven Central project structure
Update text parser to permit certain punctuation characters within tokens
Fix argument parsing for action option in ArgParser class
Added models_dir() function to urbansim/utils/misc.py to return the directory for model configuration files used by the website.
Update package name format to lowercase
Update find_and_modify() return value for upsert=True and new=True
Fix import error for unknown extensions in import environments
Added sorting functionality to ChainedSelect widget
Update Firefox service to use geckodriver and add support for CDP
Added `before` method to `OrderableManager` to allow access to the previous ordered object within the queryset.
Refactor BZ2 file handling for Python 2.6 compatibility
Refactor test_save_feed_to_filesystem method to use mock.patch decorator for mkdir function
Improve output formatting for local and publish commands
Update test_stats_service.py to use deferred result instead of None
Update setup.py to use new setuptools API for building and packaging flask_assets.
Update request schema to use strict mode
Update dependencies for moto and google-compute-engine
Added support for rad2deg and deg2rad functions in expressions
Fix tty output issue in Git wrapper library
Added CRUDEvent model and created a signal for m2m_changed events.
Fix bug in composition equality check
Update public key format for ECDSA algorithm
Improve random search algorithm to support sampling with replacement for all parameters.
Fix grammar error in resolve_term_references() function
Added support for exporting files in a zip archive format. This allows users to easily extract and share their exports.
Added support for Unicode line endings in format checker
Update API class to use dictionary items instead of iteritems() for Python 3 compatibility.
Refactoring of the transform function to use functional programming style and remove unnecessary code.
Added support for uploading packages to Red Hat's dropbox.redhat.com server.
Reorder DNS servers on interface 'Local Area Connection' if necessary
Fix incorrect increment of task try count in Task class
Add support for new EPSG codes and classes in pyepsg.py
Update celerymon to use new broker connection method
Update Stats module to handle empty stats data
Added support for using a custom state and mask placeholders in the model's step function.
Fix issue with environment variables on Windows by converting them to strings and using the correct path separator for the PATH variable.
Added required parameter for -t/--time option in arg_parse function
Fix issue with process exit code on non-POSIX platforms and Python 3
Aggregate losses and variances for each event and aggregation key
Added support for ignoring .s3conf directory in Git repository
Added support for light and thin text styles in matplotlib2tikz
Update version number to 2.2.1
Update git clone and checkout commands to use reset instead of checkout
Update CPU information parsing for BeagleBone
Update ARAnalysesField with new form values for selected analyses
Added support for non-blocking HID device operations
Update argparser fixture and test for version number
Update FieldLenField length format in RadioTap layer to '<H' from '@H' for compatibility with Scapy 2.4.3 and later versions.
Added support for synchronous query execution in graphql.py
Update requirements for compatibility with Python 3.9 and add support for sh and ujson libraries.
Update file system tree view context menu for Linux users
Added support for shared memory in StimelaJob and Recipe classes.
Update version number to 0.5.0
Added parsing for ZigBee samples data
Added support for using Infomap's `network()` method to create a network object and run the algorithm with custom parameters.
Update default domain prefix configuration for border router
Update URL patterns for pages API
Fix bug in test_groups_gather_equivalence() where stacking traces in a group resulted in multiple references to the same object.
Added support for trailing slashes in local and remote paths
[BUGFIX] Fix issue with calendar roundtrip in HDFStore (gh-8591)
Update lsb_release import and use ceph_deploy.lsb instead of ceph_deploy.util.wrappers.lsb
Added support for quoting and escaping in CSV writer and reader.
Improve memory viewing functionality by adding a new option to display data as machine words (-w) and fixing a bug in the --reverse option.
Fixing too small PoEs in disaggregation calculator by adding a check for too big PoEs and raising an error if found.
Improve SSH auto-detection by ignoring invalid input and syntax errors in device responses.
Added window title and close button hints to GoToLineDialog widget.
Update setup.py to support Python 3 and remove Python 2.7 compatibility
Modify parameters of the legislation using a modifier function.
Added support for RGBA images in locateAll function
Update version number to 0.7.7rc1
Update version number to 0.8.4
Update serialization for arrays in ipyvolume. Fix bug in update_from_js.
Update stop words for Kannada language
Update NDK version in toolchain.py to 4.8 for compatibility with Android NDK r10.
Update configuration file with new deployment information
Increased the number of positions loaded per XML file from 500 to 1000 in order to improve performance.
Fix issue with removing non-empty directories
Update to disable reloader in Flask app run() method for better performance.
Update version number to 43
Update active state of Active_QLabel widget
Added test cases for deleting categories and checking if they are not in the database after deletion.
Update version number to 2.0.4
Added support for parsing rows with multiple columns and added a new method to handle unique values in each column. Also updated the header generation logic to remove spaces and replace them with underscores.
Update sparkts test init file to use new jar name
Update input file handling in executor.py
Added version number and initialization functionality to webdriver_test_tools project
Fix issue with empty UID in read() method of jsonapi/read.py file.
Remove unused attribute 'has_ref' from Instance property.
Fix issue with ID3 tags being stored in multiple frames and not being returned in sorted order by EasyID3's get() method.
Update MNL interaction dataset generation function to include new parameter 'SAMPLE_SIZE' and fix bug in returning values.
Update install_requires in setup.py to use nionutils>=0.3.15 and niondata>=0.13.3
Added InputBase class for input parameters in table definitions
Update win_lgpo module to support new policy type
Improve performance of computing md5 for large directories by only computing it once and using a progress bar.
Update libuv revision to 93877b1
Update numpy version in setup.py to 1.16.4
Added Apache 2.0 license to anycast_healthchecker package
Fix count calculation for directories with downloads directory
Added help subcommand and custom primary command
Added support for displaying HTML logs in BuildStep class.
Refactor: Remove unused configuration options for Apex Tests DB upload
Update plot_feature_importances function to handle missing feature names
Update calculators module initialization with new development package import mechanism
Fix typo in __repr__ method of ExhaustableQueue class
Added support for CorotatingRotationWrapperPotential and GaussianAmplitudeWrapperPotential in galpy/potential.py
Update version number to 0.13.5
Update version number to 0.9.0
Fix issue with Python 2 unicode handling in UserModelTestCase
Update graph kernel computation to handle sparse matrices and improve computational efficiency.
Update Polygon WebSocket connection to use a copy of the handlers and handler symbols instead of direct references.
Added support for distributed computing using Celery for post-processing computation in classical hazard calculation.
Fix unsupported looping behaviour in UnstructuredJoin spec.
Update Prism installation script for new version
Added support for supplying a custom event loop to the run_app method
Improved template rendering performance by using Mako's built-in caching mechanism.
Update headers for JSON content type in job_listener function
Update URL for Stegano repository to use new hosting platform. Update Python version compatibility to include 3.8.
Fix issue with failed connection to Salt Master when using LocalClient
Fix issue with parsing dimensions for variables in NetCDF files. Previously, if a variable had a '_dimensions' attribute that was a dictionary, it would not be properly parsed. This commit fixes this issue by checking if the value of '_dimensions' is a list or tuple and using those values as the dimensions for the variable.
Update version number in setup.py from 0.7 to 0.9
Fix issue with reading lines from socket when using Windows line endings (CRLF).
Added a check for dangling social accounts in `SocialAccount.get_provider_account()` method to prevent errors when trying to access non-existent provider accounts.
Update setup.py with new readme and long description content type
Fix IPv6 websocket connection test for Travis CI compatibility
Fix MultiProcessValue to use process_identifier function correctly
Fix bug in ParseResults.find_in_parent() method
Added test for readpartial method of JFS object with random start and end positions
Update kernel configuration for inline figures and enable Cython magic
Update PolicyInformation class to handle policy_identifier with qualifiers
Ensure label coordinates are within image bounds before adding points to shapes.
Update HTTP protocol to handle keep-alive connections correctly
Added support for ret_config parameter in ClearFuncs class
Update GCC ARM builder to use new directory name
Added typing dependency for better type hinting support
Update AWS Batch staging S3 bucket name and ensure DynamoDB table for job tracking
Enable Python subprocesses to work with expect functionality in PopenSpawn.
Add support for new SEP character (0x01) in Hive datasets
Update charting library to round bar sizes for better visualization
Update proxy settings for aiohttp requests
Refactor LexRank algorithm to use a Markov matrix for summarization. This change allows for more efficient and accurate summarization of text.
Added support for storing and printing the number of bits used to represent each state vector in a StateVector object.
Bump version to 1.3.0 and add support for six>=1.9 in requirements.txt
Update Ejabberdctl backend to support new features and fix bugs.
Update version number in pyrogram/__init__.py from 0.13.0.develop to 0.13.0.
Added include_package_data flag to setup.py to include package data in the distribution.
Update version number and releasedate in __init__.py
Update CASClientWithSAMLV1 class to use CASClientV2 instead of CASClientBase for verifying tickets.
Fix duplicate edges in SIF file import
Added support for disabling register command for specific servers in mkrelease.py
Fix WsgiServerTransport to handle POST requests correctly
Fix parsing of multipoint WKT strings with nested parentheses
Add support for new motion sensor models SML003 and SML004
Added unit tests for Canari project using Python's unittest module and discovered all test cases in the 'canari.unittests' package.
Remove adapter and devices with a convenience method
Fix spurious dimensions of image array
Update version number in setup.py from 0.2.4 to 0.2.5
Refactor celery app initialization and add max_tasks_in_memory and max_workers_in_memory parameters
Update SPRegion class to use a random spatial pooler when learning mode is off and the SP hasn't learned anything yet.
Update Promise class to handle job store initialization correctly
Update README.rst with version number and RtD links
Update SpatialPooler class with new imports and serialization support for capnp protocol buffers.
Fix issue with reading and writing NetCDF files using different engines. (#2050)
Update instance types for SLES EC2 tests
Update hosts.allow configuration with new rules added by oinkwall v%s
Update IMAP idle support detection
Refactor AbstractIterator class to support asynchronous iteration
Update copyright year in conf.py file.
Clean up assets and remove unused files from store API.
Added CI support for test_led.py
Update InfluxDB driver to use custom port
Update Trajectory plotting function to support matplotlib version 1.1.1
Update version number to 0.2
Refactoring of transaction processing logic to use a new processor type for each family version.
Added support for secondary IDs in ReStructuredTextParser. Replaced alias IDs used in references with their corresponding primary ID.
Fix incorrect output of data_uri template tag
Fix issue with non-unicode string conversion in mediafile.py
Update ITestInterface.hello() method signature to include self parameter
Update formatting of fulltext links in bfe_fulltext.py
Added __repr__ method to _TdsSession and _TdsSocket classes to provide a more informative representation of these objects.
Improve UDP connection performance by disabling TCP_NODELAY on non-UDP connections.
Added support for FAT file system type in parted_partition module.
Update setup.py to use setuptools for packaging
Refactoring of SharePA initialization module to improve performance and reduce memory usage.
Update DocumentDetailFormView to save form data to document
Update bit length calculation for Python 3.x compatibility
Improve verbosity of CLI output by adding -vv option
Update forest graph class to use sorted edges for consistency and readability.
Update base backend class with new arguments for initialization and add parsing methods for traceback and message.
Update `modulecmd` path in ModulesEnv class
Update download URLs for Windows dependencies
Refactor internals.py to use new make_block function
Added error handling for missing 'pygments' library in syntax_color templatetag.
Update requirements for Terraform integration
Added support for Python 2.7 and 3.2-3.3 in setup.py
Fix issue with Python 3 by excluding certain files from being removed during installation.
Update dataset version by date if not provided
Update the keywords list whenever the user changes the title. This slot is not called if the title is changed programmatically.
Added support for Pipfile.lock file in heuristic filter
Update script checker to support new flags for ECDSA signatures
Added support for passing through arguments in ic() function
Renamed dewpt parameter to dewpoint in relative_humidity_from_dewpoint function
Update documentation for uibei function in main.py file. Add bounds for temperature and chemical potential parameters. Fix typo in note.
Refactor code in server/system.py to improve performance by reducing database queries and sorting BlastpHit objects by eVal
Update icon for console
Added support for Python 3.4 and above in setup.py
Added function `model_prerec_and_roc` to generate ROC and Precision / Recall diagrams for a trained model on a given test dataset.
Fix issue with missing SSL certificate and key configuration in webserver command. Add validation to ensure that both are provided together.
Fix TCP connection handling for overflow connections on OS X
Ensure SYN_DIR is available and set logging level to WARNING
Update TreeTime class to use new ancestral sequence inference method and prune short branches.
Fix unstable spilling test by reducing object size and increasing buffer length.
Update pageinfo.py to handle new filter types in PDF 2.0 (#134)
Update OAUTH2_PROVIDER_TOKEN_EXPIRES_IN to 30 days in settings.py
Fix certificate errors when copying S3 keys to Azure Blob Storage by converting the URL format
Update version number to 0.1.2
Fix content-length header for streaming responses
Update results attribute in ProblemSpec class
Update Tika version to 1.22 in tika.py
Added __del__ method to close connection and delete reference to connection object in XidDevice class.
Update print statement for running on unix domain socket or http/https protocol
Update test_fileserver.py to use new testing framework
Update BBCiPlayer plugin schema for new appStoreState field
Added support for legend labels in ElementPlot
Update version number in setup.py from 1.3.11 to 1.3.12
Added support for distributed training using TaskSpec class
Update SAML configuration files with new paths for key and certificate files.
Added error handling for dead connections and updated the connection retirement logic.
Update feed title and tagline length limit to 200 characters
Split calculation into tiles for experimental GEM support.
Split dataset into training and testing sets using random sampling.
Update setup.py to use find_packages instead of hardcoded list of packages
Added support for running tests with a virtualenv
Filling memory at %#x with %d unconstrained bytes referenced from %#x (%s)
Update tests for tif2geojson function to include new properties and ignore unknown properties
Refactor payloadBuilder method to accept a default value for format parameter
Purge installed packages and update environment
Update version number in setup.py from 0.9.24 to 0.9.25
Added support for Jinja2 template debugging in PyDev debugger plugin.
Update conformance list in test_ogcapi_features_pygeoapi
Added support for proxy configuration in request method
Update version number to 0.7.1
Fix pair trade algorithm to handle negative z-scores correctly
Added count variable to track number of rounding operations and updated roundoff error calculation.
Added support for MSVC's /LIBPATH flag and prevented whole-archive linking
Update parseProbeMatchMessage function to handle missing To element in WS-Discovery messages
Update psycopg2 version in setup.py
Added protection for local remote storage by setting read-only permissions on files and directories using shortuuid library.
Added caching for property accessor methods
Update version number to 1.6.1
Update documentation building process to support Sphinx v4
Fix import error for requests library in acceptancetests/jujupy/workloads.py
Optimize MRD model for better convergence and accuracy
Fix newline escaping in quote function
Refactor EighthSponsor queryset to use select_related instead of prefetch_related
Added tests for device property with default value and get_device_properties method when initializing a device
Update README.md and add long_description for PyPI package
Benchmark pFBA functionality and add typing for test_pfba_benchmark function
Fix TypeError in DictRegister append method
Improve performance of TokenList by removing unnecessary list.remove() operation and using a more efficient method.
Rename and describe single supplementary zip file for PoA
Refactored Jenkins tests to use new SLS file for Cloud Provider tests preparation and added support for downloading unittest reports and coverage report.
Fix command execution and logging for `py3status` module
Fix path separator in serve_package function
Add support for v1.27.1 release of Python language
Update download URL in setup.py for new version
Added support for Atom ID generation and updated the url method to use the new reverse function from Django 1.5.
Update evacuation percentage in minimum needs postprocessor
Update package version to 0.8.15
Update version number to 0.7.0
Added assertions for job IDs in JobTestCase class.
Update FlaskLoader to use DEFAULT_UNCONFIGURED_SETTINGS from _default module
Update version number to 3.8.0.dev
Update KECCAC instruction hash function to use a unique identifier for each memory location
Update API URL to use HTTPS protocol instead of HTTP.
Update version number to 1.4.11
Improved rate limiting for Bittrex API calls by adjusting sleep duration based on call rate.
Update preorder validation to account for epoch-based namespace lifetimes
Update PPO trainer to use new training buffer and reset cumulative returns since policy update.
Added support for freezing columns in VisiData sheets. The new 'freeze-col' command allows users to add a frozen copy of the current column with all cells evaluated.
Update code links in docs_resolv.py to include function names and module names
Update device model information for Osram Flex RGBW LED strip.
Update version number to 0.2.1b2
Improve FIL reader for large files by using a more efficient algorithm for calculating the number of coarse channels.
Update SheetFreqTable class to handle numeric columns with width=0 correctly
Implemented autosync for HLedger class
Improve performance of waveform plotting by reducing unnecessary baking and drawing.
Update S3 connection information for external store and cache in tests/schema_external.py
Update celeryconfig.py to set DJANGO_SETTINGS_MODULE and OQ_ENGINE_MODE environment variables
Fix zeep exception handling in TestVATIN class.
Update state highstate function to use new thin tarball format
Add support for registering resource views in SupportedServices
Fix file handling for downloading lectures
Refactor EClass instantiation to use metaclass
Added support for base units in _get_system_unit_string() method
Update binary heap implementation to use an array instead of a list
Added support for stitching multiple Z-stacks and channels
Refactor hosts_to_ignore to use set instead of list for better performance and reduced memory usage.
Update __main__.py to use Click for command line interface and YAML for configuration file parsing
Update packages excluded from pipenv
Update _APITOOLS_VERSION to '0.5.10' in setup.py
Added TestReactorMixin to EndpointMixin class for testing Endpoint subclasses with a reactor.
Fix negative scenario ID assignment in add_scenario() method
Added support for computing gradients of `scipy.linalg.solve_lyapunov` and `scipy.linalg.sqrtm` using `autograd`. This allows us to compute gradients of these functions more efficiently.
Fix encoding issue in decode() function
Update ReplicaSetConnection class to always allow queries on secondaries
Added new method 'new' to class Process in heroku/models.py to create a new process with a specific command and attach option.
Fix http://crbug.com/394454 on cros and disable tests on android and chromeos for windows.
Update version number in reana_commons/version.py from 0.6.0.dev20191210 to 0.6.0.dev20191212
Fix negative index slicing in BoundingBox class
Added test for output format inclusion in Hug API
Update py26compat.py for Python 2.7 compatibility
Update version number to 0.3.2
Update editor color scheme and highlighting for current line and cell
Update EMS API client initialization to use caching and custom transport class
Added support for caching in `newTree` function to avoid dangers of mutable cache.
Update LR scheduler for training script
Refactor instrument configuration handling in mode_run_common_obs() function
Upgrade controller to enable OS upgrade
Update pip packages and git repos in fabsetup
Added aspect ratio calculation for Beam class
Fix warnings related to datastore and PIL library
Refactor mock_raw_event function and remove unnecessary code
Added labels and help texts for AssignShiftForm fields
Trimmed EQI1 parameter to match EQB value
Fix issue with reading calculation results from datastore when multi-user mode is enabled
Update AMI protocol version parsing logic
Update worker pool to use correct number of cores
Remove unused purge_db method and update no_such_entity method
Update mock modules list in conf.py to include 'matplotlib.pyplot'
Added importlib to test dependencies and skipped tests that require fgivenx.
Update KGPotential class to include documentation for the new features and changes made in the code.
Update logging format strings for improved readability and clarity
Added support for textupdate widget in AsynOutportPart class
Update Pipfile.lock with new hashes for dependencies
Added connection_lost and error_received methods to handle socket errors and debug logs
Added support for caching zipimporters in sys.path_importer_cache
Update TeamsActivityHandler to include team information in on_teams_members_removed event handler.
Update StartWorker method to handle error response from Flink API
Update confidence interval calculation method to use exponential Greenwood formula
Ensure geohash results are included if center is given and bucket overflows
Added support for infinity values in expressions
Added try-except block for os.makedirs() in verify_env() method to handle OSError exception
Update test_reader.py to support additional hash function and task name replacement in NIDM results.
Update AWS EC2 instance termination protection for a node
Added support for asyncio child watcher to improve performance in subprocess communication
Improved MegaHAL's ability to recognize similar inputs by adding a check for similarity and updating the best score accordingly.
Update protocol mapping for TigerData cloud storage service
Upgrade master.cfg file
Fix issue with no targeted packages in setup_execute_tests.py script
Added Bitbucket URL to setup.py for easier installation and access to project repository.
Improve pagerank_weighted algorithm by using xrange instead of range for iteration number
Normalize model output for better confidence interval calculation
Refactored release_resources method to use release_resources instead of relase_resources
Fix display width of certain emojis in rich_helper.py
Added support for parsing game entity IDs in PowerLogParser
Update setup.py to use setuptools instead of distutils for Python 3 compatibility
Update deployment configuration for FreeBSD platform to include rsync and libxslt ports.
Refactoring of address parsing functions to improve performance and reduce code complexity.
Added support for disabling remote image fetching in RinohBuilder.
Update dj_database_url parsing to use new API
Fix duplicate sample names in Pangolin module
Added method `has_role` to check if a person has a specific role in a relationship.
Added support for Python 2 by using `inspect.getargspec` instead of `inspect.getfullargspec`
Fix datastore initialization in MachineCategory model
