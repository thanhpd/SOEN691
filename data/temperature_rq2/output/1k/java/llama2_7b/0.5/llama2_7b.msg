Added Map<String, Collection<String>> byCategory field in ErrorList class
Added protection for static field logger
Automatically resolving transitive class dependencies with updated configuration
Added method to get worker's address in BlockWorkerUtils
Update H2TestUtil to use a new random URL for the test database.
Added pattern for recognizing constraint violation on primary key in MongoDBDialect
Added goToInitialURL() method call in CandidateElementExtractorTest
Improved frame drop logging for ReceiverUdpChannelTransport
Added method fromStringList(String[] args) in Util class
Added private state discard logic for FLINK-23139
Update LSMSolver with GLM1_RHO hardcoded for testing purposes
Update SpringSecurityWebProcessingContextBuilder to use thread-local SecurityContextHolder

Improving efficiency of loading data in StringStringSyncTest\n
Fixed a typo in the `NodeRepresentation` class
Update rows per page in MaterialDataPager
Fixed issue with downgrading cursors under XA commit
Updating JtaTransactionManager to use transaction sync registry for rollback
Added test for non-ignored exception in commandOverridesDefaultIgnoreExceptions
Added support for JSON format in OHttpResponse
Added decoration to SearchDialogAction components for improved readability and accessibility
Reorganized capabilities directory implementation for improved performance
Fixing formatting issue in CgIsaParamTest
Added static hamcrest assertions for test methods
Added final modifier to App class for improved code quality and readability
Removed 'BIAS_FEATURE' from feature set
Update Version.java to reflect version 3.0.0-CR1
Added handling of nested city objects in CoreUnmarshaller
Update TachyonClient to use CanonicalHostName instead of HostName
Added wait for all connection closures to complete gracefully before shutting down
Fixed typos in RefTest.java
Added '|' character to replay URI for better compatibility
Added verbose mode switch for FastClasspathScanner
Removing property '{}' with value '{}' from propertyset because no matching virtual property was found
Update AbstractMockingTestCase to use mock() instead of mock(Class<T>) in order to fix a type safety issue
Update Containerless Docker Deployable Container
Fixing bug in TypeExtractor.visit() by returning inner expression instead of calling accept() recursively
Added termination of extensions before shutting down threads

Fix for Postgres regression in 1.14.0
Fixed typo in class name in XMLReportGenerator.java
Fix date format in DurationBusinessCalendarTest.java
Added alternate field for Variant class
Update VerifyBase class with user variable
Added import statement for SpinnerAdapter in Feature Demo Activity
Added quotes around sequence name for PostgreSQL
Added synchronization to avoid FWK005 parse may not be called while parsing
Added support for byte and short data types in ColumnValidationReader
Added UUID to EndpointsLock for unique identity name
Fixing view type count validation
Added new string array parameter to ConfirmTest.parameters() method
Update ApplicationClassLoader to use getClassLoader() instead of getSystemClassLoader()
Added callback for synchronous packets
Added method for setting layer rotation in radians
Fix for MongoDB collection limit query bug
Added escaping for backslash in source file comment
Added method for checking if the shell has started
Added KeyboardBehaviorTests class for testing typing behavior
Reverted changes in JNDIManager.java
Improve performance by clearing current span and setting state no tracing for sampled spans
Adjusted fraction for DeepLearningProstateTest from 0.001f to 0.0007f
Added support for ReceiveMode.PeekLock in ClientFactory
Fix for dropped frames calculation
Update AllureAttachAspects to use String.valueOf() instead of getBytes() for attachment creation
Added canRead() check for OpenJDK detection
Added 'nobs' to excluded methods in ModelMetrics class
Added support for custom node entity types
Update CharacterChromosome class with new methods and fixes
Added ability to set tick to 0 in ControllableRunner
Added finally block to catch and close client in AbstractDeployment class
Added base directory for Tomcat in UnmanagedTomcatServiceTest
Added check for listed services in TestIgnorance
Added new test cases for concatenating strings and variables
Added check for non-root directories and updated metadata for folders
Added ability to visit fragment spreads in VariableTypesMatchRule
Fix for socket client read issue
Update HibernateDaoServiceImpl to use uniqueResult() instead of listing all results
Caching optimization for DeepWater model
Added import for Call and HttpMethod classes
Fixing spark.ui.enabled for faster testing (commit #100644).
Adding sleep delay for RealDevice instances before stopping instruments
Formatted Java test file for class 'JavaFormatterTest'.
Added visibility and modifiers to myField in IClassWriter
Fix spacing for binary operations in CodePrinter.
LEGGO MY EGGO. [%s] is master.
Added import for Google Common Base Strings
Added override for TestRule when 4.9 is released
Fixed logging configuration in SiteAwareBuilder class.
Added English Word Repeat Rule and updated other rules in English language class
Added support for BigDecimal values in Proto class
Fixed styling for CmsGwtActionElement
Rewritten Es6RewriteClass to use source info from method instead of member function definition
Added Notification Channel for Upload Service
Added catch block for Exception in getChoice() method
Added PROTOCOL_VERSION and PROTOCOL_VERSION_NUMBER fields to KernelInfoHandler
Added endTag method for XML emission
Added static private variable isSteamRunning() and updated isSteamRunningNative()
Fix testExceed128Connections() to use Renjin's new connector API
Added support for retrieving process instances with `true` as the second argument in the `getProcessInstance()` method.
Added warning for multiple kubeconfig files
Removed test for SMILES format
Added support for Android platforms
Fixed AQL query formatting
Update to AbstractPatternRule.java
Update BeansXmlDescriptor to use FileDescriptor from plugin-common instead of core
Update Conf class to exclude unnecessary initialization and reload
Refactor MocoPostRequestAction to remove unused import
Added import for AttributeExpression
Added lightweight tag 'lightweight-tag-1-gb6a73ed'
Added method signature for getScheme()
Crypt class updated to fix decryption bug

Added try-catch block to handle potential NPE in shutdownNow() method
Added import statement for URISyntaxException
Added core fragments to aggregator and collector byte arrays
Update Log class to use INFO level instead of DEBUG for root logger
Added client interface handle and connection ID to MpScheduler's constructor for multi-partition initialization
Added constructor parameter for Rule and HtmlElement
Added test for inserting a null key reference in Person entity
Added caching and updating parser method in UADetectorServiceFactory
Added tax ID for CustomsDeclarationTest
Update AccessLevel enum with new value MINIMAL_ACCESS (5)
Added import for Google Guice's ImplementedBy annotation and implemented the IIndentationInformation interface with a default implementation.
Added anonymous disablement and CSFR disabling in ResourceServerConfiguration
Added FileDescriptor.setFileName() for improved plugin scanning
Added PostgresBD glue to CucumberRunner
Added assertTrue method call for mkdirs() in TestDataContextUtils
Fixed debug log statement for improved performance
Added new test class: MultiplexingTest.Muxl
Added opacity support for Raster and Vector layers in MapRendererImpl.
Added check for Docker machine status before returning IP address
Added protection for null `context` argument in `create()` method
Added support for accepting connections with a null connectUri in ClusterContext.
Added redis client name initialization in start method
Reorganized layout of ValidateFrame dialog
Added retry mechanism for remote requests
Added explicit type casting for method invocations in JavaConstantDynamicTest
Added warning for Java 8 early builds
Removed @XmlElement and @XmlSchemaType annotations for processInstanceId field in ProcessInstanceIdCommand class.
Fixed bug in RetransmitHandler class
Added column indexing error handling
Added support for empty lines in blame output
Added import statement for Serializable
Update LifecycleManagerStoreImpl to useputIfAbsent instead of put
Added a loop to iterate through the rules array in CheckConformance
Added support for LIST type in `ResolvedTypes` class
Added checkSerializable method to MojoModel class
Added listeners for 'join' and 'leave' events
Added password authentication for SSH2HostbasedConnect
Added detail to SwingTab
Update encryption classes to use new copyright year (2022)
Attaching database connection under default name
Added new method 'browseFrom' to browse index content starting from a cursor
Added import statement for Java Util List
Update ChemObject class to remove unused Hashtable import and use HashMap instead
Fixed warning message in SwingUIFacade.java
Fix for out-of-range enum value handling in DGLM class
Added test ignoring due to failure on integration server
Added ability to specify a random seed for JUnit4 tests
Fix for exception handling in ResultExporter
Added case insensitive matching for ThriftCollection enum values
Fixing blacklist path in NotificationsManager
Added assertFloatsEqualDefaultTol method for more accurate float comparison
Added leading slash to method path for consistency
Added time gauges to JHipsterMetricsEndpoint
Added DELTA variable for improved accuracy in ResizeSettings class
Added scheduling of updates for MemoryCache in MainActivity
Update Message.java with minor changes
Added support for in-place reduction of directories in PackrReduce
Update AlluxioMasterProcess with stopTimeoutVal in seconds
Improved video mosaicking with new homography estimate (commit hash)
Added validation of very large requests against a predefined max request size
Reorganize classpath split for Windows compatibility
Fixes formatting issues in PrettyLogViewerPanel
Added STATUS_TOO_MANY_REQUESTS constant for concurrency limit exceeded error messages in ConcurrencyLimitServletFilter.
Fixed copy operation for AddonRepositoryImpl
Added clearViewCached method to allow for caching of views
Update ArchiveRecordingLoadTest test duration from 20 seconds to 30 seconds.
Added cache mode local to DistributionManagerImpl
Added box and unbox methods for converting between arrays of primitives and objects
Added import statement for Android's Log class
Added rule for 'PRP_RB_NO_VB' in English language module
Added vendor support for OperaDriver
Update RadarLevel2Servlet to use Radial dataType and NEXRAD2 dataFormat
Added new method `isTrueOrNonNull` to `HamcrestCondition`.
Added load() method for function library in OMetadata
Added new target for @TA and @TB annotations in TypeVariableCycleTest
Update Jinjava class with improved caching configuration
Added if statement in consume method to handle NaN values in Sum class
Added new permissions for cluster configuration entries (CLUSTER_CONFIG_ENTRY_CREATE, CLUSTER_CONFIG_ENTRY_READ, CLUSTER_CONFIG_ENTRY_EDIT, and CLUSTER_CONFIG_ENTRY_DELETE).
Added listener parameter to ApnsConnection constructor
Added concurrent map for tracked objects
Added custom sampling method for better performance
Added handling of content for HTTP/2 requests
Updating BaseSource.java to handle empty previous work unit states
Update TimerTask to handle skipped intervals
Added support for diamond syntax in constructors, try-with-resources, strings in switch, binary literals, and multi-catch in Java Platform, Standard Edition 1.7.
Added tearDown method in X509ProcessingFilterTests
Added support for processing expressions
Added generic type parameter to CheckRange class to make it generic over the superclass
Update AsyncMultisubprocessTest to require 4 approvals instead of 2 for employee 'egonzale'
Added import for Metrics class
Added method run(conn) to RethinkDBTest
Added ClassLoader manipulation for WebXmlObserver
Added offset to ResultPoint array
Added support for reading year in R format
Added Executor for run() method
PolicyFinderModule: updated to handle serious policy evaluation failures
Update TraversalStrategyPerformanceTest to use 75% assertion percentile instead of 95%
Added offset and count parameters for terminology expansion
Fix issue #122: handle end date correctly
Added check for last available chunk id before updating
Added new test to check for resilience against spurious wake-ups
Added Vaadin form support
Updating SdkConfig with new dependencies
Update SdkUtils to use application context instead of activity context for ConnectivityManager
Fixed bug in RafMetadataReader by skipping unnecessary bytes
Added new method for starting a new span and setting it as the current span
Added reload method for transient field 'photo' in CRUDDocumentPhysicalTest
Added final modifier to method extractAsInputSource for better code quality
Added setClassName and setService methods to ProxyFactoryConfig class
Reorganized method to setResultIdx() for improved performance
Added Jackson serializers for MultipleRelationsAPI test
Fix serialization issue in CLI.java
Added method to return input field value for value components
Verify TLS session and hostname in XMPPTCPConnection
Added check for old value before replacing
Adjusted wait time in InteractiveTestJob from 5 seconds to 10 seconds.
Added matrix multiplication method for VMath class
Added support for replaying messages with query parameters
VerifyMojo: Fix for orphan attachment verification
Added check for existing project directory before copying directory.
Added method for setting coordinates in GeometricShapeDomain
Added import for AeronException
Update HelloResourcesCalendarView with styling for odd days
Added modal functionality to date picker popup in CmsDateBox.java
Added support for #many references
Adjusted ping interval and max total ping time for SimpleRoundRobinLBTest
Added extra artifacts to SingularityMesosTaskBuilder
Added tempDir variable for System.getProperty('liberty.temp.dir')
Updated Server class with new method signature
Added field for page story ID
Added import for java.util.Locale
Added waiting and timeout methods for TestCommons
Update AccountHeader class with new profile selection logic
Added query string support for redirects
Added check for null registeredService and service parameters in DefaultMultifactorAuthenticationTrustedDeviceBypassEvaluator
Fix for failed task logging
Added new ConnectionLeaseTimeThresholdExceededEventListener to FlexyPoolConfiguration
Added import statements for utility classes
Added checks for negation and inflection in PatternTestTools
Fix for possible deadlock scenario when closing ZIP output stream (SHRINKWRAP-120). Addition of log warning message.
Added check for valid invocation mode in ClusterClient
Added support for Stackdriver metric publishing
Added ROOT_PATH constant for improved performance
Added ability to retrieve request body in JEEContext
Added method for resolving resources relative to a consumer class
Added check for compatible reader type in RestRouter
Removed unnecessary wide code
Added health check requirements for Elastic Load Balancer
Modified RedisModuleMemberRegistrationWithTTLConfiguredIT test to use RedisDialect.toBytes instead of getDialect() to create a byte[] representation of the key

Added explicit owner retrieval for ATTRIBUTE_TYPE_GET_OWNERS_EXPLICIT_REQ
Added import statement for InetAddress
Fix tokenizer to use new `getTokenizingCharacters()` method
Added exports for org.tinylog.converters and provides for org.tinylog.converters.FileConverter with new implementation of GzipFileConverter
Update LZ4BlockInputStream to handle larger input buffers
Added method for getting task speed in Listener4SpeedAssistExtend
Update LiteBlockingWaitStrategy to fix race condition
Added fullRedrawHint for improved performance
NioWorker: Fix for AsynchronousCloseException issue
Fixing race condition in class definition or loading
Added sort() method to CollectorContextTest for proper sorting of context values
Added handling of whitespace before tags in PRTokeniser
Added isInputMethodTarget() method to AbstractTextViewAssert
Added final modifier to InjectorWrapper and DropwizardModule for improved code quality and readability
Added Callback field and updated caching behavior in RangeState
Added test for detecting browser version in commandsTest method
Update MVELLifeCycleManager to handle delegation of tasks properly
Added ability to split key-value pairs with spaces
Added support for correcting rect coordinates in AreaFinder instances (#100644)
Added StreamWaitTimeout parameter for improved performance when waiting for stream closure
Fix formatting of elapsed time in ThreadLog class
Added private constructor for Cycles class
Reorganized cluster executor for server stop
Added import for EdgeDriver
Hot fix for SPARQL_LANG compatibility
Fix cyclic reference in collateToSvg() for acyclic joined graph
Added UUID column and updated join column type
Added setSeparator() and setQuote() methods for customizing CSV export
Reverse List Function now uses Iterable instead of List
Fixed registration of subsystem deployment writer
Added expiry field to Cookie class
Added support for Union StreamReader type
Added support for locating classes of the boot layer with a new `ofBootLayer()` method in `ClassFileLocator`. This method returns a class file locator that can locate classes in the boot layer. (Diff committed at {HASH})
Added parameter to decide whether to create external Ingress/Routes for LoadBalancer services (default is now false)

Fixing order ID generation for HitbtcTradeServiceRaw
Added method for getting TransactionController in PersistentPriorityProcessorPool
Added WebSockets example with main method
Update CommerceAccountAddressAdminDisplayContext to fix issue with getCommerceAddressesCount() method.
Added capability for ensuring clean session in internet explorer
Added Joblog Filename to MDC

Disable waiting for Ajax attributes in repeatUntil
Added IPC channel to DriverConductor
Added checks for non-whitespace characters in TextComponent's content
Removed import statement for unused resource
Added import statement for EventListener
Update DownloadFromUrlInstaller to use getId() instead of getDownloadableId()
Fixed issue with reading chunks
Added file import for FontIconTypefaceHolder
Added charge field for isotopes in MolecularFormula class
Added array of StaxEvent types for use in ElementUtils
Added temporary path to AtomicFileWriter class
Optimized QR Code detector configuration for faster speed
Added locateCodebasesRequiredForAnalysis method to handle discovery of additional codebases for analysis
Fix for KWICPanel.trimTimeAnno() method
Update FoxPlatformDemoDataDeployer to use PlatformService for ProcessArchiveService lookup
Added check for User-Agent header and set isMobile attribute for mobile devices
Fixed BankInterruptedException in Robot class to prevent execution interruption due to old age.
Added Samsung sensor orientation type (65558)
Added put() method to workQueue in TriggerTask class
Added boolean variable SHOW_TRACE to ShellErrorHandler class
Update Config class with new values for default map entries
Modified ThreadPoolEditor class
Added endpoint field in ClusterEventDissector class
Added 3 new input types and updated existing ones in InputService.
Added method for handling flyweight tasks in QueueListener
Added check for correct resolve status in SimpleConfigList constructor
Added test for execution not to timeout if traceable with multi callback
Updating PresentsClient with minor changes (1.69 vs 1.68).
Added configuration for number of threads in LocalTestEnvironment

Implemented CONDITIONAL_EXPECTATION in RandomVariableDifferentiableAAD
Fix wrong content-type in AbstractSingleImageLayer
Reorganized print statements for better readability and maintenance
Update Display class to use WL_SEAT_EVENTS_V3 instead of WL_SEAT_EVENTS_V4
Added test for fetching a page without throwing any exception
Update Shard enum with Garena path changes
Added method to retrieve received messages without waiting
Added logging statement for LoadMetadataOptions in BaseFileSystem
Added filter to collect set of classes
Added instantiator definitions for Tuple2 class
Update ArrayProxy class with new version number
Added saturation check for ReflexivityCheckingTest
Update AsyncFile to use position() method for writing buffers
Added check for duplicate And nodes in LogicTransformer
Adding version property and method to GrailsDomainClassPersistentEntity
Fix for batch start aware in BatchEventProcessor
Added test for CertificatePinnerChainValidationTest::signersMustHaveCaBitSet
Added new static method isMainMethod to class FindOpenStream to check if the current method is the main method of the program
Increased queue size for audit log entries from 100 to 10000 in AsyncUserAccessAuditLogWriter
Added external language check in Main.java
Fixed hostname in ClientRequestExecutorPoolTest to resolve UnresolvedAddressException
Fixed error handling for non-HTTP_OK responses when uploading payloads
Update OnExpressionCondition to use new @see annotation
Reactor Core Publisher Operators: Improved Error Handling
Added field fAnnotations and updated method constructor to include it
Fixing data type of HAL response for attribute 'name' in test case 'whenExecuteExploreHALOnAttribute'
Added support for multiple place types in NearbySearchRequest
Added closing of selector and server in WebSocketServer class
Added fields() method parameter validation
Fixing missing access links in AccessSetUpdate
Improved select box functionality in BrowserTest.java
Update ProtocolConnectionImpl to use parseServerVersionStr instead of parsing a raw string
Added gossip initialization check and updated generation number display for initialized nodes.
Fix for ShaderSelectFilePropertyHandler: Removed leading 'shader/' from asset path in getLabelValue() to match actual file path
Added beacon mac address foo to EstimoteBeacons class
Fixes layout issue when there are many all-day events on a single day
Added `toString()` method with a custom string representation
Added check for allowReAssociation in ServerModel
Added check for CA atom element in HetatomImpl constructor

Added issueTime field with Unix time conversion for Livecoin data
Update profiler configuration
Added support for nulls in MapFlatSelectiveStreamReader
Added method @deprecated in DBCollection
Added sort() method to JobInstanceFetchRequest
Update JerseyServerSentEvents to use SaltStackException instead of URISyntaxException and MalformedURLException
Added reAddItem method to AddMenu class

Added support for completing node separators in operation requests
Select download peer based on chain height and protocol version
Fixing randomness in FollowerState
Update EC2 version to 2013-02-01 for AWS Cloud provider
Improved error handling in watch() method
Rollback test fix for corrupted codes
Added @DatabaseField annotation for 'id' field in class AllTypes
Fixed exception handling in KafkaSupervisor
Added ignore regex for Libnd4j assign error (2019/07/10 - https://github.com/eclipse/deeplearning4j/issues/8002). Also added ignore regex for output incorrect, can't debug b/c (2019/06/28 - https://github.com/eclipse/deeplearning4j/issues/7957).
Added new TargetType values for method return types with generic or array parameters and fields
Added file handling for next block loading
Update JDBC driver version to 1.3.5 in DriverConstants.java
Update FileSystemContext to use ConcurrentHashMapV8 for Netty channel pools instead of plain ConcurrentHashMap.
Added retry queue and updated return value for sendNotifications() method
Added method for retrieving response status
Added property 'generate.jpa.annotation' to default properties for JPA annotation generation
Fixed encoding issue in UmlautTest.java
Added support for PUT method
Added rollback protection for active transactions
Fixed classpath scan for URLPackageScanner
Added CountDownLatch for job execution
Added version number to image name in AzureOSImage class
Added method signature for WireHandler
Added print statement for special case in NewJavaFileWriter

Update ALLUXIO_DOCS_URL based on version number
Added closeable management for try-catch block
Update JPerfmonParamsDialog to handle missing colon in metric names
Added warning about drawing both in this copy and its parent graphics at the same time
Added attachment functionality to UnsavedRevision class
Fixes spell and grammar check dialog font size
Added test for empty string parsing in PathServiceTest
Fixed bug in handling variable names with '=' at the end
Added method to SavedRequest in AbstractProcessingFilterTests
Added equality and hash code methods for immature class.
Emitter thread got interrupted, shutting down.
Added import statement for RxJava
Fix deprecated method call in ViewFetcher class
Added support for Java 8's LinkedList
Added optional parameter to disable default configuration in WebSecurityConfigurerAdapter
Fixed counters reader method to use AtomicReference instead of AgentTerminationException
Added protection against NPE in createTempFile() method
Fixed insert and upsert methods to use the correct SQL syntax
Update AbstractDSLMappingEntry to use section.getSymbol() instead of toString() for better performance and consistency.
Added throw statement for internal server error when loading roles
Update JobServiceJmsIntegrationTest to increase maximum processing time to 5 seconds
Added support for onload attribute in FastArchivalUrlReplayParseEventHandler
Added reverse() method to Subunits class
Update ShutdownSignalBarrier for programmatic signaling
Added environment variable removal for Java OPTS in CommandLineInvoker
Added check for index being negative before incrementing
Added import for Jackson's JsonCreator annotation
Parsing skipped (missing type or status value in header meta data) for file {}.
Added support for loopback address via -b argument without xml files changes
Added check for getprop node
Added interceptor priority to CustomPathResolver
Update TermdBuildDriver class with improved configuration options
Fixed bug in MessageReceived constructor by adding `@SuppressWarnings(
Added import statement for PositionedException
Added check for non-existent topics
Added Eclipse patching for Lombok compatibility
Added empty constructor parameter for configuration
Fix for potential programming error in PeerGroup constructor
Added UFS attributes for root directory
Fixing incorrect element reference in CorrectEventFiringTest.java
Optimized IndexedSet implementation for better performance
Simplify handling of error codes in Finder class
Fixed loading of inner classes from InMemoryJarFile
Update Util.java with minor changes
Added check for directory existence before creating File
Fixed rendering issue in GVRViewManager
Added import statement for Mozilla JavaScript ContextFactory

Clean up worker state and stop worker
Update DefaultStompFrame class with new method toString() implementation.
Added warning for skipping getExpectedLogBelief computation in ProjDepTreeFactor
Fixed exception handling for DiscoverArchiveManifestFilesRuleProvider
Added marker with title and snippet for PlacesSampleActivity
Update Index.java to use new method for posting synonyms
Added hashCode() and equals() methods to Resource class
Added ImmutableList<PairedStats> for ALL_PAIRED_STATS
Added support for invalid Unicode characters in CsvDatastore
Added horizontal positioning error tag to GpsDirectory class
Added new poller constructor without context
Added support for spaces in URL in DefaultSpringSecurityContextSourceTests
Added import for Java Util
Added flag to indicate whether group was explicitly set to 'null'
Added thenConsume() method to ClientTest
Update SpotSceneDirector with new cluster logging
Revised TransactionOptimisticTest class with minor changes
Added static import for ConfigOptions
Update LOW_STORAGE_SPACE_THRESHOLD_PROP_NAME to match SEGMENT_FILE_LENGTH_DEFAULT
Fixed bug in write method by handling null buffer
Update Metadata class to use Servlet 2.5 version
Added try-catch block to handle multiple IE driver instances
Added optional property to GrailsDomainBinder for hibernate configuration
Update gallery configuration for CmsGalleryField
Added missing characters to word list validation
Added default group ID for FractionList
Update maximum single upload size from 5242880L to 5368709120L
Update metadata for OffHeapValueHolder
Added UPC-E expansion for searching in ProductResultParser
Update BoxedUnit to return '{}' instead of '()' for better readability
Added wait() call in runThrows method for browser execution
Added new methods for AutoServiceRegistrationProperties
Fixed serialization of Message body in MessageSerializer
Reorganized ST_Explode function to improve performance
Added check for updated drawable in LottieAnimationView's invalidateDrawable() method to ensure proper redrawing of the whole drawable when a new drawable is set.
Added a new ClientRWLock for each of the $NUM_LOCKS slots in BlockLockManager
Added ability to read the remaining of the current block
Clean up temporary builds by removing unconnected artifacts
Added method stop() to FileCacheStore
Added shared worker pool property to TargetHandler for service chaining
Added build.timestamp and build.started parameters to ExtractorUtils class.
Fixing potential dependency on ServletContext in Sham.java
Added filter to exclude messages to non-existent vertices
Updating CommerceAccountUserRelLocalServiceImpl to use local role persistence
Update DefaultSocializeProvider to fix HTTP error handling
Update AlternatingController to align checkpoints with barrier
Added checks for typed externs and goog-module whitespace wrapping
Update DefaultPerSessionLogHandler to handle larger session record sets
Added setTitleColor method to RoundedLetterView
Added empty filter check for BluetoothLeScannerCompat
Added pattern optimization for Dutch rules
Added support for starting a new line in the tail of a log file
Added sleep() call after starting MindmapsEngineServer in AbstractMindmapsEngineTest test method
Added security context check for PartitionCallableRequest
Added catch block for internal failures during authentication
Added new methods for creating super and this calls in GeneralUtils
Update ORecordAttribute to use toElement() for better performance
Update TransactionConfidence class to handle double-spending transactions
LoggingIntrospector: Print logger and handler information
Update to DefaultIndexRedirect class to fix path variable naming convention
Added OAuth signature verification for REST API requests
Removed unnecessary print statement
Added onStart() and onResume() methods to RubotoActivity
Update UserImpl class to use new canvas URL for user creation
Added instance of ChromeDriver in setUpChrome() method
Added null check for speak service before clearing it
Reorganized method signatures for improved readability
Fixed a bug in the ClientState class by removing the unused import statement for com.google.common.collect.Sets.
Added default ReleaseId for empty KieModule
Fix for issue #1234: Added missing documentation comment to Database class
Added comparison of level in compareTo method
Added definitionId field to CaseInformation class
Added import for Apache Commons Lang Exception Utils
Reorganized code for better readability
Update EsMetricsAccessorTest to use host instead of container IP address
Added notifyOnPreferenceHeaderSelected() in PreferenceActivity.java
Fix for initial fmrcDataset creation failure
Added support for creating cache config on multiple remote members
Update TableUtils: Truncate table support added for database type {databaseType}.
Added deprecation warning for closeQuietly() method
Fixed session management in SessionListenerBridge
Added fast atan2 implementation for better performance
Added subpartition strategy for HDFSDataset
Added method to check if a persistent timer should run (#100644)
Fixed byte array conversion to hex in Misc class
Reorganized method signature for improved performance
Refactoring test code to improve readability and maintainability
Fixing compiler bug in JavacProcessingEnvironment
Update DistributorFlags to allow rejecting unsupported caps immediately
Added functional interfaces for BiConsumer, BiFunction, and BiPredicate in the Throwing interface.
Update BaseClassFinderService to use ClassBundleService with corrected package name
Update ScanProgressDialog to show progress bar
Added new method `loop` in OpenALSoundPlayer class
Added publication data resource
Added NEPHELE_PREFIX constant for configuration
Reorganized code for better readability
Fixed log statement for better error handling in ReflectionUtil class (commit <HASH>).
Fixes a bug related to cleanup interval
Update AbstractInjectAnnotationProcessor to support Java 8
Fixing LSMSolver: updated code for non-SPD matrix
Fixed retry behavior for HTTP client exceptions
Fixing LrsObject import for TincanConcurrentQueueHandler
Added method signature consistency
Added mkdirs() method to create upload sub-directory
Added import statements for Jackson annotations
Added grace period parameter to TemporaryJobBuilder
Added control window and product variables for TestOperaDriver
Update Shippo class with new constants and removed DEBUG flag for production environment
Added isSupervised() method to DeepLearningModel class
Fixing screenshot comparison in NewSessionTest to use the width or height instead of both (#100644).
Added simplified consumer for executing under admin
Update CredentialReference class with volatile modification
Added isFakeBugType method to BugInstance class
Update ControlFile class with improved field handling
Update CmsSolrIndex to use OpenCms Search Manager's document factory for container pages

Update DriverPublication class to correct termId and termOffset fields
Added constructor for Section class
Update LogBufferDescriptor to use long for termLength computation
Added support for hdfs:// scheme
Added testCacheExpiration() method with ignore annotation
Added driver parameter to WebElementConverter
Added import statement for MissingResourceException
Added LinkedHashMap for property serialization
Added new SQL statement to initialize table for JDBC_PING Discovery Protocol
Fixing classpath utilities for scan classes (commit <HASH>) {classname} package name fix
Added import for internal API
Fixing buffer overflow in write() method by limiting the length of the written data
Update ConfigureForm to remove deprecated method and use new API

Added logging statement for marshalled payload string
Merged SAM files for part directory and output file
Added a new line in the IDENTIFICATION section
Added call to FlowRunnerTestUtil.startThread() to join runner thread before assertFlowStatus(), and removed unnecessary call to runner.run()
Fixing catch block to handle all exceptions
Added override for ObsoleteVersionException to avoid stack trace overhead
Added false value to VANILLA constant in Config.java
Added support for completed orders in HuobiAdapters
Added callSuper() method to Lombok's ToString interface
Fixed bug in OfflineChangeLogHistoryService by renaming file correctly
Added repeat of previous command on Enter key press
Update ServerEventTest.java with new InetAddress and host name for mirror.
Added support for removing parent block in paragraph block
Added import statement for input stream in CoverageTest.java
Added committed state to SessionSummary class
Added rounding to resampling method in Timebase class
Update DefaultScaffoldRequestHandler to include additional parameter in domain object update
Added support for Gradle command line arguments
Added sensor for monitoring CPU usage and memory usage
Added enabled state for injectable headers all requests
Updating ChannelThreadExecutorGroup with concurrent map for better performance
Optimize PathRetractionStrategy by removing unnecessary Barrier steps
Update CreateServicePage plugin with new service name
Removed deprecated methods for direct references to Servlet API
Update ProjectConverter to ask for user confirmation in case of large number of files to convert
Added resolveAppName() method to handle default app name resolution when package name is not found in launch intent
Update RealVoltDB class to expect a sync snapshot during leader election
Added default registry name check
Update JDateTime class with new time field

Added method for creating CollectionSubject instances with custom failure strategy and collection
Added method strEquals to compare strings instead of toString()
Added support for Basic authentication mechanism
Added Stun server initialization in setUp()
Added capability proxy map to NewSessionPayload
Added import for Google Guava's Throwables
Added test case for colon in path segment
Added test for null safety in ArrayPropertyUnitTests
Added import for com.arangodb.entity.ServerRole
Added sleep() method for fallback on exit
Added isRuntime() method and updated getAttributes() method to reflect changes in ThreadPoolResourceDefinition
Added Google Guava's Stopwatch for better performance monitoring
Added catch block for ArrayIndexOutOfBoundsException in Assertions.java
Added distinction between Android instrumentation test classes and directories of compiled Java classes
Added ResourceAccessor to BaseLiquibaseTask
Added method signature for ResourceConfigApplier interface
Fixed charset in Loader.loadFile to use StandardCharsets.UTF_8 instead of Charset.forName("UTF-8") for better compatibility and performance.
Added exception handling for missing reference in Postconditions
Added attribute list to BodyCells model
Added template and asset folder monitoring
Update logging level from INFO to DEBUG in LoggingTest test class.
Added check for directory writability before overwriting DSO file
Update PublicationImage class to handle liveness timeout for image connections
Fixed bug where leaving the room would throw an exception if occupant information was not reset first
Fixing issue with subitems counting in FastAdapter class
PinListSync updated with new pin list from worker
Fixed encoding of canonical string for AWS S3 PING request
Update RawType class with Apache License v2.0
Update CloseableAdapter to fix compile error
Update DefaultFileSystemMaster class with ConcurrentHashMap
Fixed potential runtime exception in LraFilterComponent
Added visibility check for hidden nodes in NodeInstanceImpl
Added option to specify external bitext XML rule file
Added handling for 3-byte surrogates and 6-byte surrogate pairs in StringSerializer
Fixed exception handling in DatastoreImpl
Added type safety for getCount() method
Warn users about compiling from project files (deprecated) and fix typo in `success` variable name
Fixed permission check for locale group creation
Added method for scripting support
Fixed NullPointerException in JRubyAdapter
Fixed bug in Elasticsearch import service to avoid 'All shards failed' for big datasets. Added flush call to ensure data is properly indexed.
Added environment variable 'cluster.routing.allocation.disk.threshold_enabled' to Elasticsearch container
Added null checks and removed unused variables in ByteSelectiveStreamReader
Added logging for middleware classes
Fix escape level import for compatibility with Java 17+
Added waitAllForSafeState() call after creating new Hazelcast instance
Added default nonce generation based on time
Added extended parser class functionality
Update PresentsServer to handle repeat shutdown requests
Fix for IllegalStateException when closing JDBC connection
Update SurfaceLayerGL with final field
Fixing date format in FramerManager tick method
Added support for handling multiple response headers
Added DLSequence object to ECKey constructor
Added role-based security constraints for Undertow runtime
Added early termination for properties
Added check for null or closed file system before creating new one
Added chance for listener to modify mojo configuration before execute
Fixed validation for www.metaescort.com in VerifierServletTransaction
Trim whitespace and newlines from StoryState output
Added test for getInt1() method in MockTest class
Added copy replacement function to BranchTemplateService
Added closing of output writer in `delombok.Delombok` class
Added finally block to catch exceptions thrown by class statement in RepeatTests class
Updating PlaceRegistry with new mapping
Adjusted period calculation for correct year counting
Added optimization option to RegisteredLookupExtractionFn
Added runtime stage for ServerRemoveHandler
Added check for varying lengths of share values
Added CAMEL_SPLITTER_PATTERN and camelSplit/camelPhrase methods to utilize it
Added LTHeadMenu and ContextMenuInterceptor for debug mode
Added List<DeterministicKeyChain> field in KeyChainGroup class
Added constructor for LegacyAddress
Fixed ignore exception in class initialization
Added import for AuthoritiesConstants
Fix for InstructionSet.java: added null check in isDeclaration() method to prevent NPE
Added toString() method for BootChargingStationCommand
Added support for FileDatastore in JsfHelper
Added a new line of code to handle empty option sets in HelpFlags.java
Added ToolStrip for zoom and save buttons
Added runPreCrawlingPlugins method in CrawljaxController
Added mkdirs check before writing to file
Added getFirstHeaderValue and getFirstHeader methods to BasicHttpClient class
Added support for writing packet bodies using PacketManager.writePkg() instead of PacketManager.write()
Added delay to system exit (2*1000ms)
Added support for collapsing title bar background color
Fixed memory registration for NATRegMrCall
Update Activity.java to reflect changes from 100644
Added final modifier to emit() method for better code quality
Added new test case for ORDER BY with MP subquery
Added serialization of sequence number for CommonLogFields
Added protocol check for URL strings in BoxBrowser class
Update CanvasTestBase class with new constants for OAuth token and connect/read timeouts
Added method for blocking until next notification is available
Added Locale support for column names in DatabaseValidator
Fixing disposal of Disposable in DisposableHelper
Streams: updated methods to close output streams
Improved Bootstrap logic in Setup.java
Update to use static RxJavaObservableExecutionHook for consistency
Shuffling all rows in DRF (#100644).
Added support for handling large result sets in FindAllQuery
Added method to load classes without checking for false positives
Added logging for icon URL
Increased send and receive buffers for unicast socket to handle large state transfers
Added support for multiple platform versions in AndroidSdk
Update AndroidServlet to handle StaleElementReferenceException with a better error code (10 instead of 13)
Added check for model class in ModelUtils
Update ExchangeCookieUtils to use milliseconds instead of seconds for maxAge calculation
Added constructor for JAXBUtil
Added base URL check for BrowserMobProxyTest
Added bindings for text pane width and line positions
Added suppress warnings for unchecked cast in ClassContextTest.java
Added transaction rule builder
Added loading of activejdbc.properties from classpath
Added import statement for @Local annotation
Added terminateToken field for consistency
Added phenotype validator to Engine class
Update Update.java to handle multipart/mixed media type
Refreshing Kafka client configuration after modification
Added support for nested expressions in LazyPatternBy
Added/Edited Properties in WebAppDatabasePopulator.java

Added randomness to generated ID for improved performance
Added test for Jmol animation
Added import for Collections
Added indirect field in MutableBySettingFieldOfField
Update ChannelSecurityConfigurerTests to use Jakarta HTTP APIs
Added HGVS code generation for SNVs
Added m2repo directory in ArchetypeTest
Fixed path matching for S3 resource resolution
Added boolean variable for tracking whether the site has received the first fragment task message
Added support for children in ModelObjectWithChildren
Improve performance by reducing unnecessary gap handling
Added type parameter <ULong> to method <T>channel</T>Map<ULong>.get() to match updated signature
Added support for referencing schemas in ChangeLog output\n\nChanges:\n\tAdded a new method `add` to the `DiffToChangeLog` class to handle references to schemas in the ChangeLog output\n\tModified the `tabName` variable to include the schema name when generating the ChangeLog output\n\tFixed a bug where the schema name was not being properly escaped in the ChangeLog output
Fixes encoding of non-BMP code points in DatabaseUtils
Added support for customizable ripple color and type in RippleView.
Added method for getting contents with line endings
Added new method for processing TSMeta through trees
Added ability to obtain builder instance through RepositoryService
Added test for none rows other thread exact two queries other
Added generator method for starting traversal
Update List class to use OListElement.TAG or UListElement.TAG for constructors
Fixed issue with outstanding messages
FileStoreLogFactory: Low storage warning threshold reached (usableSpace=
Fixed toggle method to click instead of setSelected() for consistency with other selenium tests
Updating Gradle versions in GradleCompatibilityExtension
Added SPARQL service description namespace and updated managed namespaces
Update Utils class to return correct value for heap size calculation
Added support for batched statements in LoadDataChange
Added support for transient modifiers in AdapterUtil
Added parameter handling for MetricParams in PerfMonCollector
Added natural string representation orderings to UsingToStringOrdering
Added support for disabling global AST transformations
Revised TestMultiThread class to use a fixed thread pool of 100 instead of 200 for better performance in multi-threaded tests.
Update Engine class to handle remote link close/detach events
Fixed test timeout for TestHiveRecoverableGroupedExecution.testCreateBucketedTable to 60 seconds
Added global lock for bitmap decoding to prevent excessive memory thrashing and OOMs
Added `not()` method to Predicates class
Added 4 transitions to TransitionSet class
Added column list recursion check in Index class
Added support for handling events with body in FiniteStateMachine
Fixes CI build issue for RSS and Atom tests
Fixing drain() issue in SoundManager
Added implementation of LombokFileObjects class
Update test cases for LocalDate factory methods
Added constructor for UserProfile with ID, name, firstName, lastName, email, and username parameters
Fixed compatibility issue with newest evolution engine API
Added import statement for @Ignore annotation
Added import statement for com.hazelcast.test.annotation.QuickTest
Update ExclusiveBufferClaim to use byte instead of short for flags
Added support for Boolean parameter in OptimalCECPParameters
Added import statement for Objects class
Added synchronization to makeThreadLocalLoopers()
Added method toString() to return path including file name
Added report content functionality
Added implicit flag encoder mapping for GraphHopper
Update SiteMojo class with forward slashes
Update to ClientDelegatingFuture class
Added import for SelenideElement and updated method to handle it
Added method for storing and retrieving rule data
Update PlanPrinter to include CPU time metrics
Added constructor for CloudSpannerOAuthUtil class
Added SQL formatting for Create Table, Select, Truncate and Drop Table commands
Added import for URI class
Added weak hash map for slave reconnection
Fixed random number generation for output file name
Configured evolution engine to execute Weasel program
Added missing resources in ModelTest
Update BCrypt class to use constant-time comparison for salt and hash values\n\nThe `rawSalt` and `rawHash` fields were swapped in the previous version. This commit fixes that by using `Bytes.wrap()` to wrap the raw data and compare it using a constant-time algorithm.
Added mapping for node type in DocumentGraphMapper
Added JCommander parse method for bug patterns
Update ProxyDefaultMethodHandle class to fix invokeWithArguments method
Added underscored ISO code for I18n class in Java\n\nChanges:\n\t+ Added method `getUnderscoredIsoCode()` to return an underscored version of the ISO code for a given locale\n\t+ Moved the `lexer()` method to a separate class, and updated the qualified name to include the locale name in uppercase\n\t+ Fixed typos in the `map()` method
Added check for type parameter in Check class
Fixed loop in ClassFileWriter for JDK7 compatibility
Added logging for different packet versions in MulticastService
Added list-based implementation for checking if a DAG has running jobs
Fixed shutdown hook for NodeService
Added LinkedIn ID format string to Identity.Simple constructor
Update WorkerOptions class with new default value for workers
Added exception skipping feature in FormatterMojo
Added upgrade() method to lift security restriction on underlying channel
Added support for Bidirectional and OutBound transactions in Main.java
Added support for culprits in RecipientProvider
Added check for AudioDisconnectEvent before closing audio connection
Flipping buffer for readObject() in InstallRequest
Added import for Google Common Preconditions
Fixing incorrect end position in SourceCode class (#100644)
Adding support for large maps on Windows
Added list of partitions for constraint enforcement
Update LakeBTCExchange with new SSL URI
Update DemoApplication with new connection timeout and read timeout, and add commit() at the end of the constructor
Added warning for RefreshListener when no ServiceMatcher is provided
Added exception handling for report method in ScheduledReporter class
Fixes bug in JLanguageTool.java by updating lastLineBreakPosInError and lineBreaksToEndOfError fields
Added isIdElement method for proper element determination in PairBasedRestfulRouter.
Added check for underscores in method signature
Added WebSocketListener to WebSocketSubscriptionTransport
Added ability to disable properties through newInstance method
Fixed error handling in FileList.java by adding cancel extra for when no input file name is provided.
Added method to create adapter for preference screen
Fix for sleep interruption in SQL batch (#100755).
Added import statements for Jackson annotations
Added support for escaping non-ASCII characters in CSV output
Update BasePairParameters class with new parameters
Added support for parsing field as double
Simplify and minimise automaton for Type class
Added import for ArtifactType and updated default artifact type in SvnConnectorConfiguration
Added TreeSet for command name storage in CommandManager
Added support for empty type variables
Added new exception classes for cache result unavailability and corrupted content
Added 'messenger_extensions' and 'fallback_url' properties to WebButton class
Added parameter 'parent' to readItem method
Update Discovery Entry Store InMemory to trace capability found instead of debugging
Update to AbstractRepository to use Spring SpecificationsExecutor for query counting
Update to StandardTitanGraph for better performance
Added ingress publication for AeronCluster client
Modified CFMetaData class to add a check for invalid index name changes
Added test for covtype GBM
Added name field to GeoJsonObject class
Added conductorCycleThresholdNs configuration
Added import for RejectedExecutionException
Added domain-users.properties file to security scan
Fixing bug in Task.java by removing unused line and updating taskStateTracker registration
Fixing compiler bug in JavacProcessingEnvironment
Added displayName and picture fields for GitHub provider
Added method for recycling ByteBuf
Added field 'comment' to Mediawiki API request
Fixed test for Circuit Breaker open exception when error occurs not on subscribe
Update RepackageMojo to use latest artifact handling API (fixes #100644).
Added support for MongoDB 4.x GridFS API
Added documentation for J2ObjCIncompatible annotation
Added method for creating metadata with no assigned ID
Fix sigmoid implementation for numerical stability
Update SslTest to use TLSv1.2 with TLS_DHE_RSA_WITH_AES_256_GCM_SHA384 for MariaDB testing
Added and updated word replacements for GermanSpellerRule
Fixed unpackInt method to read byte from input stream correctly
Added VelocityEngine and updated filters for DevelopmentPlugin
Update PlannerBenchmarkConfig to use AUTO threads
Added managed attribute for setting slow and very slow invocation thresholds in milliseconds
Reorganized code for better efficiency
Fix for CDK fingerprinter: added upto search depth in path calculation
Added manual provider check
Added unzip method for zip files
Added missing `size()` method call in `SingleUserResult` class to match changed behavior of EnterpriseExtension API
Added email and subject extra data to DialogRateMe Intent for sending mail
Added support for entity <T extends Configuration>
Added support for including x-amz-acl header in PutObject requests
Fixed scroll behavior in MapController
Added support for boolean type in XbaseTypeComputer
Fixed loop invariant assumption in VerificationTransformer
Added registerPlugins() method and updated WebView configuration
Added version number to Transaction class
Fixing bug in JemmyDSL class by removing unnecessary System.out.println() call
Modifying ExceptionProxy class to fix constructor call issue
Added forward decls for private locals only
Added warmUpPartitions method to SmallClusterTest
Added singleton schedule option
Added dependency changes from previous successful build
Fixed imports for WarJSFIntegrationTest
Updating StreamSource.java to schedule latency marking with fixed delay instead of rate.
Optimized Selector: com.hazelcast.internal.networking.nonblocking.SelectorOptimizer
Fixed stacktrace in TestUtil.java
Added alt text for image list editor in DropDownImageListEditor class
Added nodeRole field to Cluster class for disk usage statistics
Added support for resolving type arguments using the `resolvedTypes` field in AbstractBeanDefinitionBuilder
Added --provision flag to VagrantState enum for provisioning during run.
Added import statement for Notification class
Recursive comparison of Avro schemas for record fields
Update AbstractRemoteCommandObject class to handle error response from REST API more robustly
Added brand file for book locale
Added closing of server socket after process destruction for better resource management
Increase property field prefix length
Added method override for toString() in FluxMaterialize class
Added configuration monitoring in AutoReloadingConfiguration
Fixing error in creating AgiScript instances of type class
Fixed typos and updated import statement
Added 6 characters to the constructor of `PUBREC`
Added test for large date format
Collapse Properties Test Fixes Compiler Convention Print
Added production mode check based on MSCLayer standalone status
Added note on inconsistency in retrieve behavior and changed retrieve to return profile type instead of base type.
Added support for appSecurity-5.0 feature set in GrpcServerComponent
Added IBM_UTC_ID to TimeZones class
Increase chunk size for file scanning
Added support for monolingual corpus and parallel corpus in LanguageTranslation class
Update to PostgreSQL 9.0 compatibility
Clear out expired listeners from right to left to avoid shifting index issues
Added print statement to assume that features meet requirements
Fixing StrictMode policy in MainActivity to detect all potential errors and log them for debugging purposes.
Restoring Snapshot for path: {path} with nonce: {nonce} and dupsPath: {dupsPath}. (Time taken: {timeTaken}ms).
Added location field to ArtifactQuery class
Update DiskLruCache to handle edited entries correctly
Update AsyncScalr class with new thread pool configuration
Update FxmlLoadResult to remove unnecessary logger
Added support for last chunks of CaddScore data
Added import statement and method for building all KieBases in a KieModule
Added clear up all attachments method in AbstractDeploymentUnitService
Fixed bug in removeAll method
Added a new repository for JSON schema validation tests
Added serialization support for LazilyLoadedCtor
Added candidate term ID syncing in Election class
Added option to disable summary generation
Added defaultGroupId and defaultQualityGateUuid fields
Added check for URL changes before waiting for opera to be idle
Added method to convert string to Date
Fixed null pointer exception in Script.java by updating the `exec` method to return the result directly instead of using a try-catch block.
Fixed bug related to partially constructed object access in PartiallyConstructedObjectAccess.java
Added tree generation for AnalyzerPhase
Added index check for corrupted hint names
Improve IFFImageWriter performance by buffering last 10% of image data and writing it in one go
Added path field to ResourcePersister model for non-library resources (SONAR-4245)
Update OGlobalConfiguration to use a new limit of 1024 open files simultaneously
Added check for writability before calling write() to handle backpressure
Update ShardingProperties class to handle incorrect input types
Update AndPermission class to use Context instead of Fragment
Added assertion for acquired locks in PidRuntimeFactoryTest
Renamed srcInode parent id and name in MasterInfo::writeToDisk()
Added cache for access check in Permit
Update PBDSegment to handle truncated buffers
Revised SqlStageExecution implementation to exit cleanly upon query cancellation
Fixed output of CfDef object
Added check for client closure in ExchangeOperator
Update TreeNode.java - Add method onLongClick(TreeNode node, Object value) and change return type of onLongClick from void to boolean
Added validation for empty 'problems' list in ConfigException
Added configuration file for ServerIntegrationTestBase
Fixed issue with importing unnecessary dependencies.
Added binding for PriorityResource class in SingularityResourceModule
Added check for new Response object in FilterContext.java
Added token counter for better error handling
Improved performance of OneShotBenchmark by reducing slice from 4 to 20
Added support for DB2 Database in AddPrimaryKeyGenerator
Added nextNextIndex to log buffer descriptor
Added support for specifying a static constructor name for Data class
Added check for operation mode before allowing snapshots in JOINING mode
Added JspcMojoClassLoader for compilation with tools.jar (#100644).
Fixed debug logging for sending messages
Added compareAndSet method for atomic sequence update
Fixed nullability of inserted student in ModelTest.java
Fixed test for PeriodicalRefreshStrategy to verify at least 2 refreshes instead of 3.
Added new method for checking if process is still running
Added method 'updateUserInformation()' in SearchUI class
Added explicit type cast for method receiver in DefiniteAssignment
Added business central documentation check in Kie Controller Swagger Integration Test
Update safe mode logic for worker connect wait time
Added test for encoding special characters in PDF417 codes
Added validation for timeslot duration
Revised NettyTransportTest to use longer timeouts for tests
Added empty check for access token in MPRestClient::getAccessToken
Added attributes to include in VCF output for CmdLineAnnotatorUtils
Added check for available partition writers before returning true from isRunning() method
Added waitForClusterToForm() call for multiple cache managers
Added support for empty type parameter lists in BeanSpec
Added test for empty performance state container
Layer refactoring
ResolveExpressionTypesPass: Fixing unknown or error types in field arguments
Added getPolicyRuleURL() method for PolicyEvent
Added X-Text-Delivered header to log responses
Added method for handling request permissions result in NativeScriptActivity
Added native method for interrupted() check
Added error message for PORT_CONFLICT error code
Update maxTextLength limit in ApiV2
Added deprecation warnings for putAll and keySet methods
Fixed issue with newline and tab characters in URI normalization
Fixed whitespace issues in XHTMLWhitespaceXMLFilter
