Added method asMap() to ErrorList class to return a Map<String, Collection<String>> that represents the errors grouped by category.
Update Logger instance in StatsdLoggerImpl class
Automatically resolving transitive class dependencies with the following configuration: <config>debug(
Added utilities for tachyon.worker.block package
Update H2TestUtil to use a random seed for database connection URL
Update MongoDBDialect primary key constraint violation pattern for compatibility with Fongo
Added crawler initialization and sleep method for better performance and added exception handling for CrawljaxException when initializing the crawler.
Update receiver UDP channel transport to log frame lengths correctly. Fixes #1234.
Fix: Update Util.java with new fromStringList() method
Discard private state in ChangelogStateHandleStreamImpl (FLINK-23139)
Implemented adaptive Rho calculation for ADMMSolver, temporarily hardcoded to GLM1 value for testing purposes.
Update SpringSecurityWebProcessingContextBuilder to use MODE_THREADLOCAL for SecurityContextHolder
Added constructors for creating JedisShardInfo with name and weight parameters.
Improve loading initial data in StringStringSyncTest
Remove unused OutputSymbol class from NodeRepresentation.java.
Added methods for loading and unloading data from MaterialDataPager component. Added method for reloading data.
Added test for downgrading hold cursors under XA transaction.
Fix issue with JtaTransactionManager not properly rolling back transactions
Update BasicDefaultIgnoreExceptionsTest to use new exception types
Refactor OHttpResponse to use a custom format provider for rendering response content.
Refactored search dialog to use captions for better readability and aesthetics.
Refactoring of capabilities directory implementation for server-side capabilities querying using a concurrent hash map instead of an in-memory database.
Update CgIsaParamTest to compare results with pretty-printed expected result
Update WaitStrategyTestUtil with new assertThat method for testing
Refactoring: Removed unnecessary import and reduced code size by using a final class for App
Update feature extractor to use simple features instead of complex ones
Update management version to 3.0.0
Added support for nested CityObjects in CoreUnmarshaller.
Added support for getting local worker host name using getCanonicalHostName() instead of getHostName().
Refactor ClientConnectionsShutdown to simplify and improve code readability. Removed unused variable 'closePromise' and added a comment explaining its usage. Modified the logic for waiting for connections to close gracefully to ensure proper shutdown of client connections.
Update reference tracking code in RefTest.java
Added support for replay channels with multiple destinations by appending | between them and ? at the end.
Added verbose mode for debugging purposes. Enable it with the `verbose()` method.
Refactoring of VirtualType class: removed debug logs for no matching virtual properties
Fix issue with mocking test case in xwiki-commons-test
Added support for resolving main cube when deploying a Containerless Docker Cube
Fix a bug in TypeExtractor where inner expressions were not properly handled
Added support for loading and initializing extensions at runtime, now users can add new features to Jill with ease.
Added new test cases for array initialization
Update DatabaseContainerFactory to support PostgreSQL and Oracle
Update test suite name in XML report generator to GraphWalker (GraphWalker instead of GraohWalker)
Update DurationBusinessCalendarTest to use SimpleDateFormat with the correct pattern (MM instead of M)
Update Variant model to handle missing values in alternate and reference columns of the VCF file.
Update user for Maven repo.
Refactoring: Removed unnecessary imports and updated Java code formatting.
Fix issue with PostgreSQL sequence names containing single quotes
Fix issue with XML validation in STIXSchema.java (#<HASH>) by synchronizing the validator and re-throwing the SAXException (#<HASH>) for better error handling.
Added support for Byte and Short data types in ColumnValidationReader to handle various schema formats.
Added a new example of how to use the Java API for endpoints locking
Added support for multiple view types in Recycler.
Added support for Russian names in confirm() method (#234). Fixed issue with unstable Chrome browser.
Refactoring of ApplicationClassLoader to use a dedicated classloader for application resources
Added support for sync and async packet listeners using a PacketFilter. This allows developers to specify which types of packets they want to receive notifications for. #XMPPConnection
Update rotation method to support new origin
Added support for limit and skip in MongoDBCollection#find with options argument
Update JavaWriter to handle escaped backslashes in source file path
Added methods to check if REPL is running and started. 
Added KeyboardBehaviorTests class for testing the keyboard behavior of StyledTextArea
Refactor JNDIManager to use rebind instead of bind
Refactor ZipkinSpanListener to clear current span before sending a new span to Zipkin server.
Update DeepLearningProstateTest to reduce runtime
Added public access modifier to ClientFactory class
Update FPS calculation for dropped frame recording in AnimationFragment class
Refactoring of AttachAspects class. Changing type of result parameter from byte[] to String and adjusting charset encoding to UTF-8 in order to improve performance and fix issue with incorrect attachment display.
Added a check for whether the tools.jar file exists and can be read by the agent.
Added support for new model metrics nbobs (number of observations)
Added support for retrieving schema information for both nodes and relationships.
Update CharacterChromosome class with new method and refactoring
Fix issue with resetting the tick value to 0 after a successful run
Fix bug in AbstractDeployment class: add finally block for client safe closing
Added support for setting Tomcat's base directory to `target/tomcat-$simple_name` in UnmanagedTomcatServiceTest
Fix: Add missing condition for ignored test case check
Added tests for string concatenation and variables in arrays. Addresses issue #345.
Added support for creating metadata for folders when listing objects
Added support for visiting fragment spreads in the VariableTypesMatchRule class.
Improve performance of SocketClient by reducing number of socket operations
Fix bug in getNextId method. Previously, an unchecked warning was suppressed and the list was checked for size instead of using a uniqueResult.
Enable data caching for DeepWater model training
Added support for GET requests in RequestAuthenticator extension.
[CI] Update Spark version in AnchoredPredictorTest to disable UI
Added sleep for 1500ms before killing instruments process on RealDevice to avoid issues with iOS device
Update JavaFormatterTest to use new version of JavaFormatter (782bb452c7080a7e0bd2eed57d4d2ce2b7febaa3 instead of c178cea8d43d16a4741dffc9be4e1b0a2931e897)
Added support for annotations and properties in fields declarations in IClassWriter interface.
Refactor CodePrinter to improve code readability and maintainability
LEGGO MY EGGO. [%s] is master.
Added support for percentile aggregation in pivot search result series.
Fix: remove unnecessary override of TestRule's `apply` method to fix build error
Removed @Slf4j annotation from SiteAwareBuilder class
Added new rule for repeated words in English language (EnglishWordRepeatRule.class)
Update type and value for boolean values in Proto class
Update CmsGwtActionElement to use XHTML syntax for <meta> tag
Fix incorrect source info for static method assignments in Es6RewriteClass.java
Disable sound for upload notifications
Fix bug in getChoice() method of AutoCompleteComboBox class
Update protocol version to 5.1 in KernelInfoHandler
Fix a bug in XmlEmit where a closing tag was not emitted correctly. Add new method endTag() for ending a tag. (#<HASH>) (CC@<USERNAME>) (<REVIEWER>)
Added support for checking if Steam is running and added a flag to indicate if it is running.
Fix connection limit in Renjin128ConnTest.java
Update process instance status and add active signals in response to an update signal event
Fixed warning in ClientBuilder when loading multiple kubeconfigs from $KUBECONFIG env variable
Refactor ReaderFactoryTest to use new PDB and SMILES format objects instead of hardcoded strings.
Add support for Android in Conscrypt SSLEngine check.
Update query builder with new AQL support
Refactor AbstractPatternRule class to make it more modular and easier to extend. Modified methods include isGroupsOrUnification(), isGetUnified(), isSentStart(), isTestUnification(), getPatternElements().
Update BeansXmlDescriptor to use FileDescriptor from Common plugin
Removed unnecessary initialization of RapidoidInitializer in Conf static initializer.
Refactor MocoPostRequestAction to use Guava's ByteStreams instead of Apache HTTP Client's InputStreamEntity
Added support for attribute expressions in GeneralUtils class.
Update test for DescribeCommand with lightweight tag support #502
Update AbstractTFS class to support new TFS scheme.
Fix bug in decryption logic by updating InputStream reading loop
Added support for different class loaders and class loader isolation in CacheManager lifecycle management
Fix NPE in ExecutorShutdownUtil.shutdownNow() method of org.jboss.netty.util package.
Added support for reading and writing ModeShape repositories using file system resources.
Update aggregator and collector fragments in AdHocPlannedStatement for improved performance
Update log level from DEBUG to INFO in Log4J configuration.
Added connectionId parameter to constructor of MpProcedureTask to support sending invocation messages to the correct client session.
Fix: Remove unnecessary blank line in Violation constructor
Added tests for inserting and selecting null key references
Rename method 'getCachingAndUpdatingParserHolder' to 'getCachingAndUpdatingParser'
Added EIN and EORI number for exporter identification
Refactor AccessLevel enum to include new level MINIMAL_ACCESS.
Added implementation of IIndentationInformation.Default
Disable anonymous authentication and disable CSRF protection for resource server
Added check for non-null path before setting it as the file descriptor's filename.
Fix typo in PostgreSQL glue path
Update TestDataContextUtils class for testing data context substitution
Fix performance issue in DataSetTableIterator.java by optimizing the retrieval of data from the database.
Added new test cases for Muxl and Demuxl classes
Fixes for setting the opacity of vector layers
Start container using DockerService
Refactoring: create() method now has protected visibility and returns a Connectivity instance instead of void
Added support for accepting connections from external hosts
Update RedisAppender to use a JedisPool with configurable buffer size
Update layout of ValidateFrame dialog
Improve retry mechanism for remote requests by adding timeouts and handling TimeoutExceptions properly.
Update JavaConstantDynamicTest to fix failing test
Update ByteBuddy mock maker to ensure it only creates non-private & non-final classes
Refactor ProcessInstanceIdCommand to make it more readable and easy to understand.
Fix for incorrect counter increment in RetransmitHandler
Improved error handling for inserting and removing columns.
Fix for SONARPLUGINS-3097 - Git do not report blame on last empty line
Added support for serializing the object. Fixes #123456789.
Remove unnecessary code and optimize performance
Added support for checking multiple conformance rules in a single iteration
Added support for custom allowable values for enums using the "LIST" format
Fix serialization issue with MojoModel class
Added event listeners for group membership changes and added an execute listener.
Update SSH2HostbasedConnect.java to use password authentication with hostbased connect
Refactoring: Moved list of visible pages and details from `History` to `SwingTab`, removed duplicate code.
Update copyright notice for AO Industries, Inc.
Update PreparedStatement close method to use new closePreparedStatement method instead of closeBatch
Refactor: Rename method 'browseFrow' to 'browseFrom' to align with latest API naming conventions. Add deprecated tag for old method name.
Added support for running segment metadata queries in parallel by wrapping Sequences.toList() around the input sequence. This allows the query to execute more quickly and use less memory.
Removed unnecessary Hashtable import and replaced it with HashMap to optimize memory usage and improve performance.
Update SwingUIFacade to use JOptionPane for confirmation dialogue.
Fix issue with enum values not seen in training data.
Fix reloading of anonymous types in test suite.
Update JUnit4 task to support user-defined random seed override. Fixes #123.
Added try-catch block to handle exceptions when exporting charts and added a mechanism to reset image name if an exception occurs
Update ThriftCollection enum to use lowercase names
Refactor NotificationsManager to use Guava CacheBuilder and remove JetBrains annotations
Added JUnit assertion for floating-point equality checks with a default tolerance value of 0.000001f.
Added support for paths that start with multiple slashes
Added support for process metrics using `TimeGauge`.
Added delta for comparing float values in ResizeSettings constructor, fixed negative quality value check.
Schedule updates when memory cache is set
Update Message class to support negative header IDs
Refactoring: Improve Jar handling and support for in-place packaging by supporting directories as input
Update Thrift RPC server stop timeout configuration
Update image motion estimation to improve performance and accuracy. Increase the number of pyramid levels from 200 to 220 to improve speed and accuracy of the algorithm. Adjust the parameters for the tracker and the homography model to improve performance.
Added validation for very large requests. See <HASH>..<HASH> for details.
Refactor classpath string splitting to handle Windows drive letters properly and support basic splitting on ':' under all operating systems.
Added support for displaying log messages with custom simtime and action columns.
Added support for HTTP status code 429 in ConcurrencyLimitServletFilter when throttling is triggered.
Fix copy directory for addons
Added method to clear cache of view retrieved.
Update test duration for ArchiveRecordingLoadTest to 30 seconds.
Improved performance of write operations by skipping remote lookup and enabling cache mode local in InterceptorChain
Added utility methods for converting between int arrays and Integer array lists (#523)
Added support for printing stack traces of exceptions with Logger
Fix issue with PRP_RB_NO_VB rule causing false positives
Remove unused dependencies and simplify imports
Update dataType for RadarLevel2Servlet to use 'Radial' instead of 'Station', as required by NEXRAD2 format.
Refactor HamcrestCondition to use Callable instead of raw item
Reload metadata from disk
Added support for type variable cycles in generic type annotations. Fixed a bug where type variables were not correctly checked for cycles. This update improves the handling of type annotations and makes the compiler more robust.
Optimize Jinjava expression factory for better performance by setting cache size to 0 and disabling expression caching.
Fix NaN issue in Sum node list
Added new permissions for cluster configuration entries: `clusterconfigentry:create`, `clusterconfigentry:read`, `clusterconfigentry:edit`, and `clusterconfigentry:delete`.
Added null check for listener in ApnsConnection constructor.
Added concurrent support for tracking weak references of objects
Update proposal distribution for linear regression model to use custom GaussianProposalDistribution with learning rate of 0.1
Added support for handling chunked transfer encoding in frontend HTTP connections when proxying requests to backend servers. This fixes an issue where non-chunked clients were not able to send content to the backend.
Update latest watermark based on previous work unit states if present. If no previous work unit state is found, set the watermark to default value.
Fix timer task scheduling issue
Add support for Java SE 7 and 8 language features in SourceVersion enum
Refactoring: Removed unnecessary code and improved code readability. (#<HASH>)
Fix issue with double processing of expressions in ExpressionAnalyzer. Fixes #123456789.
Fix type parameter bounds for generic CheckRange predicate.
Update approvalsRequired property in AsyncMultisubprocessTest to match new business logic
Added support for disabling Micrometer metrics collection in Reactor Netty clients and servers.
Added support for creating a new database using RethinkDB's dbCreate method
Fix NPE when using OSGi classloader as context classloader in WebXmlObserver.java
Update GenericMultipleBarcodeReader to handle null ResultPoint objects in decoded barcodes.
Update JapaneseElementTest to fix twoDigitYearFormat() test failures caused by JDK-8065557. (#134)
Added support for custom executor in `Async.run` method
Improved performance of PolicyFinderModule by reducing the number of schema parsings performed in `evaluatePolicy` method
Update performance threshold for TraversalStrategyPerformanceTest
Fix bug in expansion with count = 0 by returning immediately if count is 0
Added code to handle end date with seconds issue in Range.parse
Fix issue with reverted last available chunk id in GameUpdateTask.
Added tests for Pool.claim(long, TimeUnit) with a timeout. Added tests for null ttl-unit and null allocator. Also added comment on racy nature of test case.
Added support for setting and getting the Vaadin form that contains the AbstractFormPropertyRenderer. This allows the renderer to be used in different contexts and makes it easier to integrate with other forms.
Update dependencies for Robolectric's SdkConfig class to use new artifact versions
Update connectivity check to use application context instead of system service
Fix bug in RafMetadataReader where the for loop was going out of bounds. This change fixes a potential null pointer exception that could occur if the image file contained unexpected data.
Added support for starting a new span and setting it as the current span using the `startSpan()` method of the SpanBuilder class.
Fix transient field behavior in CRUDDocumentPhysicalTest
Added final modifier to method signature in XmlExtractorHelper
Added a no-args constructor and getter/setters for service in ProxyFactoryConfig
Update Tree class to use new getRow method
Fix issue with SimpleDateFormat in MultipleRelationsAPITest
Fix serialization issue with Arrays.asList in CLI class.
Implemented getText() method for ValueComponentHelper class to return the correct value for boolean fields
Fix hostname verification issue in XMPPTCPConnection.
Added support for returning the old value in a successful replace operation
Added new functionality to InteractiveTestJob to allow for a different waiting period than the default of 5 seconds.
Added a new method for linear algebraic matrix multiplication, equivalent to timesTranspose with second vector as a matrix.
Fix incorrect calculation of termId in ArchiveConductor#initReplayForExistingRecording() method.
Fix orphan documents verification in VerifyMojo
Added project creation mechanism
Update GeometricShapeDomain class to fix bug related to setting coordinates and update geometry.
Fixes a bug in LogReplication where stopping a log replication could cause an exception to be thrown.
Remove unnecessary hour height setting for DayView in HelloResourcesCalendarView
Added modal popup for date picker
Fix #many references not being resolved properly in BatchLinkingService.java
Update ping interval and max total ping time for load balancer.
Update SingularityMesosTaskBuilder to handle empty URIs list and new extraArtifacts field
Update RunLocalPayloadTest to use tempDir for creating a local Liberty runtime
Fix logging configuration for production environment
Added support for retrieving page story ID for published photos
Fix issue with loadNativeLibrary on non-Linux systems
Added methods for waiting and timeout for tests (#321)
Update AccountHeader to support selectionList and fire event for onAccountHeaderListener
Fix redirect behavior for requests with query string
Update DefaultMultifactorAuthenticationTrustedDeviceBypassEvaluator to handle null registered service and service objects
Fix: Fix error handling in TimeScheduler3
Added support for connection lease time threshold exceeded event listener
Added support for empty query results (null pointer dereference fix)
Fix false positive detection of exceptions in rule pattern test tools
[SHRINKWRAP-120] Possible deadlock scenario: Got exception on closing the ZIP out stream: 
Fix for #12345 - Invalid invocation of PackagedProgram.execute() method
Update StackdriverMeterRegistry to use new Micrometer API.
Update Contents view to navigate to root path instead of current selected path when changing workspace
Added support for reading request body in JEEContext#getRequestContent()
Refactoring of ResourceResolver class. Resolves resources relative to a consumer class if they are not absolute paths. Fixes #123456.
Added support for multiple error handlers and writers.
Fix wide instruction encoding in ClassFileWriter.
Update ElasticLoadBalancerCapabilities to remove unnecessary method and add new one
Fix RedisModuleMemberRegistrationWithTTLConfiguredIT test for Redis dialect change in Hibernate OGM 5.3.7.14 and above.
Improve performance of DefaultChannelId by reducing the length of the String representation of the channel ID
Added support for explicit attribute owner retrieval
Update RestHttpHelperTest to use InetAddress for hostname
Update tokenization characters in WordTokenizer
Added support for Java 9 module exports and provides declarations in module-info.java file.
Update decompression buffer size for better performance
Update Listener4SpeedAssistExtend to support calculating task and block speeds
Fix deadlock issue in LiteBlockingWaitStrategy by adding a lock to avoid race conditions when waiting for the cursor sequence to advance.
Update back and front buffers when screen size changes, set fullRedrawHint to true
Fix: Handle AsynchronousCloseException when reading from socket channel
Fix race condition in class definition or loading
Fix sorting of list in collector context test.
Fix whitespace handling in PRTokeniser
Added method to assert that a TextInputView is set as the input method target.
Refactor: Simplify Platform class by introducing a new DropwizardModule and using an InjectorWrapper instance instead of a raw instance. Removed unnecessary fields and methods.
Refactor callback interface to use functional interface instead of interface reference
Fix bug in BrowserVersionDetectionTest#commandsTest()
Fix issue with incorrect task status after delegating
Refactor Parser class to improve performance by optimizing string split operation and reducing memory allocation
Added support for Locale-sensitive string comparisons in the `AreaFinder` class. The property `GALEN_BROWSER_PAGELEMENT_AREAFINDER` was added to set the default value of the `AreaFinder` field, and the method `correctedRect` was modified to use a locale-sensitive string comparison.
Fix issue with waiting for closed streams
Update ThreadLog class to use nanoseconds instead of milliseconds for performance optimization.
Refactoring of Cycles class to use a private constructor for instantiation and adding a new parameter for edge-to-bond map.
Update ClusterExecutor initialization for better security handling
Added ignore annotation to EdgeTest class to avoid test execution failure due to missing Microsoft WebDriver binaries.
Temporary hot fix for comparing two strings with different languages. Replaced with !eq if either of the predicates is not a string in XSD.
Refactoring of collate() method to generate acyclic joined graphs instead of cyclic, unjoined graphs.
Added support for storing UUIDs of push messages and their associated variants in the database.
Refactored CSVExporter to allow for customizing the separator and quote characters using setters. Added setSeparator and setQuote methods to provide this functionality.
Fix reverse method to use flatMapIterable instead of flatMap to avoid unchecked warning
Refactor ApplicationClientConfigurationPersister to simplify code and remove unnecessary methods
Added support for cookie expiration date and setter for it.
Add support for union and decimal types in StreamReaders
Update ClassFileLocator to support locating classes from the boot layer. #1
[DeployMojo] Update parameter for creating external Ingress/Routes based on exposecontroller's decision. #1625 (#1625)
Initialize IOSServer class
Update order ID generation method to support client-side ordering IDs
Added method to return transaction controller for database transactions.
Added a comment to explain how to run WebSocketsExample.
Refactor CommerceAccountAddressAdminDisplayContext to improve performance by reducing unnecessary database queries
Added support for ensuring clean session in Internet Explorer launching using `CapabilityType.ForSeleniumServer.ENSURING_CLEAN_SESSION` capability.
Update LoggingAfterJobListener to use MDC for logging file name
Fix: Use `this` instead of `model` in the `extractAndTransform` method to avoid NoModelExtractorException when passing a custom template engine adapter.
Added support for disabling ajax attribute checking during repeatUntil(). Resolves #1234.
Update Subscriber registration for IPC channel
Added support for compacting text components with only blank characters
Removed unused dependency on javax.annotation.Resource
Added support for registering event listeners and filters
Update downloadable ID for DownloadFromUrlInstaller
Refactor SnappyInputStream to use a larger buffer for reading compressed chunks
Added support for initializing FontIconTypefaceHolder with a File object instead of just an AssetManager and String path.
Update MolecularFormula class to handle nullable charges correctly.
Update StaxEvent enum with new events and improve event handling code.
Refactoring of AtomicFileWriter class to use NIO.2's Files API for atomic file writes. This change makes the class more efficient and resilient in case of power outages or other unexpected events.
Added fast QR Code detection configuration to improve speed and reduce computational complexity of the detector.
Added support for discovering additional codebases containing classes required for analysis
Refactor KWICPanel.trimTimeAnno to make it more readable and avoid unnecessary code.
Update deployment of Fox Platform demo data
Added support for mobile devices by detecting user agents and updating the theme accordingly.
Added check for old age and added exception for interrupted execution due to bank interrupt.
Added support for Samsung devices with their proprietary sensor orientation type (65558).
Fixes issue with commit log being disabled, preventing trigger task from running
Fixes #1234: Adds support for boolean `SHELL_ERRORS` variable in ShellErrorHandler
Added support for simultaneous duplicate queries (#647)
Refactor ThreadPoolEditor to remove unused fields and improve code structure
Update ClusterEventDissector to include follower member ID and log position information
Added try-catch block to handle IOException and APIException when getting input types information in InputService class.
Added support for flyweight tasks in QueueListener
Refactor SimpleConfigList constructor to ensure correct resolve status
Fix issue with IsStillRunningServiceTest that was causing executions to timeout for long-running operations. Added SlowTest category for this test.
Fix stale connections issue with resumeSession method
Update LocalTestEnvironment to use a customizable root folder for local runtime
Added a simple implementation of hashCode() method in UnsafeBuffer class.
Added support for conditional expectation estimation in RandomVariableDifferentiableAAD class
Fix wrong content-type handling in AbstractSingleImageLayer. Throw a RuntimeException if failOnError is set to true, otherwise create an error image.
Improve performance of BlackScholesDeltaHedgedPortfolioTest by reducing number of simulation paths and print less statistics.
Update seat proxy binding for V3 and V4 WlSeatEvents interfaces
Added a new test case for handling a URL parameter and updated the Main class to handle command line arguments properly
Update Shard enum to use garena paths and urls
Added method to retrieve received messages without waiting. (Refactored) #1234
Added loadMetadata method with LoadMetadataOptions parameter for better customization
Refactor: Migrate from deprecated Class#forName to Class.forName
Refactor: update TupleClassMeta to use new constructor syntax
Update ArrayProxy version to 3.1
Fix: Add missing initialization of reflexive and non-reflexive properties in ReflexivityCheckingTest
Update AsyncFile to use newDirectByteBuffer for direct memory allocation
Refactoring of the LogicTransformer class, added duplicate removal for And nodes
Added support for versioning of Grails domain classes. (#<HASH>)
Refactor BatchEventProcessor to use waitFor() method in SequenceBarrier to ensure that the batch is started only after all events are available.
Added check for CA bit in signers' certificates to ensure that the chain is validated correctly.
Update FindOpenStream detector to handle main methods
Increased the queue size for async user access audit log writing to reduce the risk of logs being dropped due to full queue.
Fix bug in I18nLanguage comparison method to also check external status
Test non-existent host connection
Fixed issue with error handling in SegmentHTTPApi.
Refactoring of OnExpressionCondition class in org.springframework.boot.autoconfigure.condition package. Changes include updating imports and renaming class name to OnExpressionCondition.
Refactor: Update Operators.java to handle RejectedExecutionExceptions by wrapping and logging them instead of throwing them.
Refactor Description class to include new parameters
Refactored HALBuilderTest to use new get; statement for exploration queries
Added support for multiple place types in NearbySearchRequest#type method
Fixes #1234: Update selector implementation to use a new thread for handling client connections.
Added validation for passing fields in pairs as name and value when calling field() method of ODocument class.
Update Access Set for Object Type: 
Refactor select option functionality to improve performance by using an optimized method for selecting options. The new implementation uses a single click() call instead of two.
Added caching for parsed server version string
Update NodeCmd to print gossip initialization status and fix generation number output when not initialized
Update ShaderSelectFilePropertyHandler to use shorter shader path for label display.
Update BeaconConnection constructor to support new beacon class
Update header height calculation to prevent overflow when too many all-day events are displayed. Fixes #1234.
Added `toString()` method to PostmarkResponse class to provide a human-readable representation of the object
Fix for issue #1234: Update ServerModel class to handle re-association of HttpContexts with bundles correctly
Added support for searching atoms by full name in HetatomImpl
Improve SSL connection handling in ConnectionFactoryImpl
Update LivecoinAdapters to support new data types (#<HASH>) @<USERNAME>
Update profiler configuration with elapsed time to create snapshot and maximum snapshots and summaries
Added support for creating DictionaryBlock with null mask and RunLengthEncodedBlock in MapFlatSelectiveStreamReader.createBlockFromKeyValue() method.
Remove deprecated methods for adding fields to a document before saving it to a collection.
Added support for unsorted paging in Batch2CoordinatorIT.
Fix issue with proxy configuration for Jersey client.
Re-add item functionality for AddMenu component
Fix bug in LockFreeCache.java where a checkClosed() call was missing after locking the cache.
Added support for completing operation names and address selectors using a new method in OperationRequestCompleter class
Added a new method 'selectDownloadPeer(List<Peer> peers)' in the PeerGroup class which is used to select a peer from a list of Peers to be used as the download peer.
Update FollowerState to reset last voted for candidate and adjust election timeout
Update EC2 version for Eucalyptus compatibility
Refactored watch method to use a separate thread for long-running operations
Fix issue with many corrupted codes in text by implementing rollback on timeout
Added generated ID field to `AllTypes` class to test `refresh()` method.
Added a new try-catch block to catch Throwable instead of Exception
Fix assign error in xlnet_cased_L-24_H-1024_A-16 model and remove cifar10_gan_85 from test suite due to output mismatch.
Refactor TargetType enum to include new targets for method parameters and fields.
Fix issue with reading files from an iterator in BlockFileLoader.java
Update JDBC driver version to 1.3.5
Remove unused import and fix propagation of InterruptedException in acquireNettyChannelPool
Added functionality to return all unsent notifications from both the main queue and retry queue in a single list. This change aims to improve the efficiency of the queue management for clients that need to keep track of all unsent notifications.
Added methods for accessing response status and type in ResponseWrapper class.
Added support for generating JPA annotations with generate.jpa.annotation property set to true.
Fix encoding issue in UmlautTest.java
Added support for PUT requests
Fix rollback of transaction in ObjectifyImpl.java to handle failure and log error
Refactor URLPackageScanner to improve performance by replacing String substring and replace methods with a single String replace call
Added support for multiple jobs running concurrently and tested the concurrency behavior of JobManager.
Update AzureOSImage class to set image name and description based on providerMachineImageId
Added support for custom wire handlers
Update NewJavaFileWriter to skip unnecessary type checks
Remove unused `useNewByteCodeGenerator` field from CompilerConfig
Update ALLUXIO_DOCS_URL constant to reflect current version
Added support for closing additional closeable resources in the case of an error during the execution of a codec block.
Fix performance issue with JPerfmonParamsDialog component by initializing all metric elements in the constructor
Added warning about using multiple graphics objects at the same time
Added support for adding attachments to unsaved revisions
Added a new font size for the language label in the SpellAndGrammarCheckDialog, increased from 12 to 14.
Fix parsing of empty string in PathService.parsePath().
Remove unnecessary code and simplify shell initialization
Update AbstractProcessingFilterTests.java to handle GET requests correctly.
Added detection for classes without equals/hashCode methods and annotated with @ImmatureClass, which is missing the default implementation.
Fix thread interruption handling in Emitter class of AsyncWaitOperator to prevent unexpected shutdown.
Update EmptyInventory to support RxJava Observables
Refactoring: Remove unnecessary suppression of deprecated warnings in `getScrollListWindowHeight` method.
Added support for using a Deque as the implementation of the ImportStack instead of a Stack.
Disable default configuration in WebSecurityConfigurerAdapter.
Refactoring: Removed unnecessary import of java.io.PrintStream and method printCounters() from ClusterTests class.
Added method for creating temporary files and updated URL to use https protocol instead of http.
Update EmployeeDAO methods to include a throwing clause for the SQLException
Update AbstractDSLMappingEntry to use new getSymbol method for section symbols instead of hardcoded strings.
Added error handling for loading roles in UsersResource.java
Increased maximum processing time for job service tests to increase test coverage and stability.
Added support for onload event handling in FastArchivalUrlReplayParseEventHandler
Update Subunits class to handle large stoichiometry values more efficiently. Fixes #1234.
Refactor ShutdownSignalBarrier to use a shared static latch for better performance and consistency.
Fix JAVA_OPTS environment variable for child processes started by CommandLineInvoker
Added check for negative index in RoundRobinExecuteWithFailover.java
Added support for creating GrokPattern objects using Jackson annotation
Fix parsing of meta data from header (missing type or status value).
Added support for loopback addresses in interface criteria parsing.
Fix bug in NodeUtil.getRootNode() by ensuring that the passed node is actually a GetProp node.
Added alternative and priority annotation to CustomPathResolver
Update build driver configuration to use internal network by default
Update MessageUtils.java to include @SuppressWarnings("unchecked") for deserializing a JSON object into a MessageReceived object
Fix exception handling in Exec.java
Update KafkaSupervisor to handle missing topics
Patched Eclipse APT Processor to handle newer versions of Lombok's injected code
Update ClusteredObservationBus to use supplied configuration for instantiating the JChannel instance
Fix potential programming error by checking for attached block chain before starting PeerGroup
Added UFS attributes to web interface general servlet
Update CorrectEventFiringTest to use By.id instead of By.tagName
Remove unnecessary logger and import statement from IndexedSet.java.
Refactor Finder class to simplify and reduce the number of exceptions thrown.
Remove unused assert statement in VoltCompiler
Update Util.java to use new ClassLoader instance
Added directory creation logic for DnsCacheFile
Refactor cullAndRender() to remove dependency on postEffectShaderManager and postEffectRenderTextureB.
Added support for using multiple contexts in structr-core.
Added support for decimal data type mapping in JDBC Type Mapper
Stop UFS in LocalTachyonCluster
Update DefaultStompFrame class to include missing method 'toString' and change return type of 'content' variable from 'ByteBuf' to 'String'. (#<HASH>)' to include a reference to the specific commit hash. This will help others quickly understand the changes made in the commit. The updated message should be: 
Improve performance of getExpectedLogBelief computation by skipping the computation if it has already been done and returning a pre-computed value instead.
Fix exception handling in DiscoverArchiveManifestFilesRuleProvider
Added a new marker for each place in the Places fragment's map
Update synonyms API endpoint to use `JSONArray` for input and output
Added hashCode and equals methods to Resource class
Update StatsTesting with immutable lists of statistics and paired statistics
Added support for reading and writing CSV files with UTF-16 encoding
Update GpsDirectory tag names to use consistent language
Refactor poller() method to deprecate use of ZMQ.Context or ZContext.createPoller
Added support for spaces in LDAP URLs, as well as improved error handling and documentation. Thanks @[Author Name]
Added test for retrieving cells with timestamps in correct order.
Fix issue with GroupProperty.get() when using nullValuesOnly parameter. Added a new flag to indicate whether all values are null or if a group was explicitly set to "null". This fix resolves a bug in the behavior of the method and ensures that it works as expected.
Improve performance of ClientTest by using new consume methods
Fix issue with cluster change detection and resetting old cluster object.
Fix transaction optimistic cache management issue in tests (#897)
Remove unnecessary description for metric options
Update default low storage space threshold for archive to 24 segments
Fix NPE when writing end of stream with empty buffer and null write promise.
Update default servlet version in Metadata class from 2.4 to 2.5.
Fix: canStartMultipleIeDriverInstances test fails due to multiple drivers quitting, now uses try-finally block to ensure both drivers quit properly
Fix issue with optional one-to-many relationships. Set collection role and value for referenced entity. Fix optional property for many-to-one relationship.
Update gallery configuration and locale handling in CmsGalleryField.
Update word list for Catalan and Greek alphabetical characters
Added default group ID for fraction list.
Update maximum single upload size in SingularityS3UploaderConfiguration
Update OffHeapValueHolder to flush metadata changes after update to ensure data consistency
Fix UPC-E parsing issue
Update BoxedUnit.toString() method to return {} instead of ()
Added support for browser testing using Envjs.
Added support for enabling and disabling AutoServiceRegistration with new 'enabled' property and deprecated 'registerManagement' property.
Update serialization of Message header and body fields
Update ST_Explode function to find first geometry column and its SRID
Added support for invalidating a specific drawable instead of always invalidating the whole root drawable. This is necessary because the animation may be too large to fit into memory and need to be broken up into smaller drawables.
Improve performance of BlockLockManager by pre-allocating ClientRWLocks.
Added support for reading the remaining of the current block in FileInStream.
Remove temporary builds from built artifacts and delete unconnected artifacts.
Improve performance of file-based cache store by using a custom executor service for writing and stopping the service during shutdown. Fixes #1234.
Refactor TargetHandler to support service chaining
Update build start time and timestamp in ExtractorUtils class
Fix issue with ImagePicker not properly initialized when used in Servlet environment.
Added a filter to remove MessagePayload instances in the computation of the SparkExecutor. This is needed because if there are messages that point to non-existent vertices, they will not be processed correctly.
Added support for fetching roles from `RoleLocalService` and updated code to use it instead of `RolePersistence`. (#1234)
Remove debug logging statements from DefaultSocializeProvider class.
Refactoring of AlternatingController to improve performance by reducing state machine complexity and memory usage.
Added support for generating Closure goog.* scope aliases in typeless code.
Refactoring of DefaultPerSessionLogHandler to optimize performance by using a more efficient data structure for storing session logs.
Added support for setting title color attribute through setter method
Added support for filtering ScanResults based on user-provided filters
Fixes issue with case insensitive regex token for Dutch patterns
Improve performance of PerLineTailSplitter by using a dedicated buffer to read lines
Update test class to start MindmapsEngineServer
Added support for security context in PartitionCallableRequest
Improved error handling for authentication providers to account for internal errors and improve failure tolerance
Added methods for creating constructor calls and declarations with explicit targets for super and this classes. (#143)
Fix: Update ORecordAttribute to use Optional and Streams instead of Raw Records
Added support for handling dead transactions in TransactionConfidence class
Update logging introspector to print logger and handler information instead of all other information
Added support for custom index redirect
Added OAuth support for Flickr4Java REST calls by signing requests with an access token and secret when available.
Removed unnecessary println statement in SomeSteps constructor
Update RubotoActivity.java to call onStart() and onResume() in run() method
Update user creation URL to include account ID and fix broken API call
Update setup method for ChromeDriver
Fix memory leak issue by removing the speak service reference from the place object
Added support for runtime browser selection based on system property `selenium.defaultBrowser`. Fixed bug in `setTestContext()` method.
Remove unused Guava dependency and simplify set retrieval
Update ClasspathKieProject to use ReleaseId from maven pom properties file or container's default ReleaseId if not found
Refactor: Migrate from Java Collection Framework to Google Guava's Cache library for improved performance
Added compareTo method for SingularityTaskMetadata class
Added support for defining a definition ID for cases in ProcessContextTest
Improved error handling for metric expressions evaluation. Added a new counter to track datapoint reads.
Refactor ToggleSwitchBase to use new style of setting sizes and colors. Removed unnecessary lines.
Update Elasticsearch client configuration for test
Added a new method to notify listeners when a preference header is selected.
Fix fmrcDataset creation failure
Added support for creating a cache config on remote members with the createAlsoOnOthers flag set to true.
Remove unused variable 'connection' from clearTable method
Remove deprecated method `Closeables.closeQuietly()` and update implementation to better handle exceptions.
Fix issue with SessionListenerBridge not properly cleaning up when tearing down a servlet request context
Improve performance of atan2() by using a faster approximation method provided by Math.stackexchange
Added support for creating HDFSDataset objects using a builder pattern.
Fix issue with non-persistent timers not being executed correctly.
Refactoring: Replaced StringBuffer with StringBuilder in Misc.java
Improve performance of interval tree search by rearranging traversal order
Refactoring: removed unused methods from ReflectTest class
Fix a compiler bug by adding a comment to suppress an unused variable warning.
Update DistributorFlags to use new config format and add default value for reject-unsupported-caps flag
Added functional interfaces for checked exceptions to the Throwing interface and its subinterfaces, including Predicate, BiConsumer, BiFunction, and BiPredicate.
Added support for finding classes in packages with a trailing /
Improved performance of ScanProgressDialog by using SwingUtilities.invokeLater()
Added new methods for looping sounds with different gain and position settings. (#692)
Update resources for SurveyorCore
Added new configuration key 'NEPHELE::' for Nephele-specific parameters.
Update thread dump format to include double quotes for thread name
Fix bug in getDomainClass method of ReflectionUtil class.
Optimize Whirlpool's cleanup interval for improved performance by using a mask instead of a fixed value.
Update supported source version to Java 18
Improved LSMSolver to handle l1 penalties and upper/lower bounds more efficiently.
Fix authentication failure handling in DefaultClientAdaptorImpl for HTTPClientException with status code 401
Refactor TincanConcurrentQueueHandler to use a more modern concurrent data structure, such as ConcurrentLinkedQueue.
Refactor NamedSubResourceSettingBuilder to use new API
Create upload sub-directory under server temporary directory
Added support for serializing and deserializing ErrorMessage objects with Jackson JSON processor
Added grace period option to temporary job builder. (#4243)
Update TestOperaDriver to support new platform and product types.
Update DEBUG flag in Shippo class. Set value to false.
Added isSupervised() method to check if the DeepLearningModel is supervised or not.
Fix issue with screenshot width and height for new iOS driver session.
Added a new method `sudo` that allows execution of code under admin privileges. This is needed for certainOrientDB operations that require administrative permissions.
Update credential reference attribute definition and add support for destroyable interface.
Added support for missing and found bug types
Refactor user-defined fields in ControlFile class to use modern Java practices and remove unnecessary code.
Added support for indexing container pages in Solr index
Remove unused dependencies from WarTestProbeBuilderImpl class.
Fix data frame heartbeat header length check
Added constructor for Section class to allow for setting of member variables
Update LogBufferDescriptor.java to align termLength and PARTITION_COUNT with a long type
Added error handling for URI schemes with malformed syntax.
Update TitanEventualGraphTest to ignore cache expiration test due to concurrency issues
Added support for caching remote web elements in Selenium web driver wrapper. This change ensures that our custom caching web elements are used instead of the original ones from Selenium, which helps improve performance by reducing the number of HTTP requests made to the browser.
Fix for missing resource exception in ConstantsWithLookupProxy
Added LinkedHashMap to preserve field / property order for json / xml / yaml serialization purposes.
Added log statements for JDBC_PING Discovery Protocol to handle exception thrown during table creation. Added logging level debug to avoid printing stacktrace by default.
Refactored ClasspathUtil class to support scanning classes for a custom root package
Update test dependencies and add AlwaysIncreasingSystem2 for DB testing.
Fix: corrected error in Utils.java
Update deprecated method signature for isSubscibe(). Use isSubscribe() instead.
Added Utils and Records as reserved keywords for generating Java code.
Improved logging for marshalling SOAP payloads.
Fix: check for _SUCCESS file before merging SAM parts (#42)
Added a check for functions with no outputs in SimpleParameterList#getParameterCount(), defaulting to 1 if necessary
Fix issue with FlowRunnerTest2 not properly running flow executions in parallel
Fix AssertionError handling in JSpecRunner test execution
Added method to avoid overhead of stack trace for ObsoleteVersionException class in versioning module.
Update Config.java to enable Turkic Azerbaijani case folding
Added filter for completed orders in `adaptOpenOrders` method.
Update ToString annotation to include superclass's toString implementation by default
Refactor OfflineChangeLogHistoryService to use a more efficient method of deleting and renaming the history file
Added support for arrow key navigation of command history in GUIConsole class.
Added support for customizing hostname in Syslog4J server events
Refactor ParserUtils to optimize parsing of text blocks.
Added support for loading classes from a list of files
Update session summary status enum and rename 'performing' status to 'committing'
Added rounding of resampling result
Added support for updating domain objects with a custom callback method.
Fixes issue with running Gradle CLI command when specifying a build file. Previously, the command was not working correctly when passing a path to a specific build file. This fix resolves the issue by using the correct syntax for running the command.
Refactoring of MonitoringAgentService: replaced CpuUsageProbe with CpuUsageSensor and MemoryUsageSensor in start() method
Added a new checkbox for enabling or disabling the scan of request headers when using the 'Injectable Headers' option in Active Scan.
Update ChannelThreadExecutorGroup to use ConcurrentHashMap for executor services instead of HashMap.
Added a new step (NoOpBarrierStep) after EmptyStep in a traversal that is not a child of MatchStep and has a Barrier before it to prevent redundant path compression.
Update default service name in create service wizard to mySample
Refactor: Deprecate getHttpServletRequest and getHttpServletResponse in Context
Refactoring ProjectConverter class for better support of large projects. Ask for concrete matcher implementation and user confirmation in case of large number of files to convert.
Update RhodesService to support custom app name resolution.
Update leader appointer initialization for sync snapshot support in REPLICA mode
Added a new method `SharedHealthCheckRegistries.getOrCreate(name)` and a new method `SharedHealthCheckRegistries.tryGetDefault()` to allow getting or creating a health check registry by name and returning null if the default has not been set, respectively. This change is part of the Metrics Healthchecks project.
Refactor JDateTime.hashCode() to use new `time` field instead of `jdate`. This change is necessary because `time` is now used in a more optimal way than `jdate`.
Update nullability annotations for ElementUtil class.
Added static factory method for creating a CollectionSubject instance with a custom FailureStrategy and collection.
Update toStringEquals() method to use strEquals() instead. This change is necessary because the new code uses a different algorithm to compare two strings.
Update auth mechanism mapping for BASIC authentication
Update TestWithStun class to use STUN server for public address discovery.
Added support for proxy configuration in NewSessionPayload.java
Refactor DefaultLogicResult to use Google Guava's Throwables for throwing exceptions and remove unnecessary code.
Added test case for handling of URL suffixes containing colons in query parameters
Added null safety checks for ArrayProperty constructor and assertArrayType methods. Prevents NPE when passed a null PropertyDescriptor.
Added support for renaming views in single server deployments with ArangoDB version 3.4 or higher.
Fix infinite loop in CliTableauResultView.java that caused client to crash when CTRL+C is pressed.
Added isRuntime method to ThreadPoolResourceDefinition class to determine whether the extension is runtime or not.
Update dependencies for BalancePartitionsStatistics.java
Improve JS script element validation by handling array index out of bounds exceptions.
Update AndroidTestFinder to find test classes compatible with InstrumentationTestRunner
Added support for using a custom ResourceAccessor in BaseLiquibaseTask subclasses. This allows for more flexibility in how resources are accessed during Liquibase operations.
Refactor ResourceConfigApplier interface to make it more consistent and easier to use
Update charset for loading files from disk to use `StandardCharsets.UTF_8` instead of `Charset.forName("UTF-8")`.
Fix: Update Postconditions to throw IllegalStateException when an assertion error is not thrown.
Added support for new attribute `attributes` in BodyCells class
Update BakeWatcher to watch for changes in content, template, and asset folders
Update log level in LoggingTest class from 'INFO' to 'debug'.
Fix issue with unpacking SOs by deleting and re-extracting if necessary. This fixes a bug that caused SOs to become corrupted on some devices.
Update active transport count in publication image metadata based on subscriber positions and liveness timeout
Refactoring of MultiUserChat class. Removed redundant lines of code for userHasLeft() method and added a new comment explaining the purpose of the changes.
Update FastAdapter to handle new items in a better way.
Update PinListSync to retry on failure
Update S3_PING protocol to use HTTPS instead of HTTP
Update copyright year for RawType class.
Update CloseableAdapter to use new onClose method signature.
Update dependency versions and fix concurrent hashmap initialization issues in DefaultFileSystemMaster class.
Update LraFilterComponent to register serverside side filters using CDI current instance select and get methods
Added support for hiding nodes based on metadata in `NodeInstanceImpl`. Check if node is hidden and remove it if necessary.
Added support for using external bitext XML rule file and activating neural network based rules with --bitextrules and --word2vecmodel options. Additionally, updated documentation for these features.
Fix a bug in StringSerializer where surrogate characters were not handled properly.
Fix issue with getting changes from datastore - propagate illegal state exception correctly
Update CountCommandResult to use Number instead of Double for count
Deprecate compiling from project files (replace with new command line syntax)
Fix issue with resource locking in CmsLocaleGroupService
Refactoring of PredefinedQuery class: removed jdk.nashorn.api.scripting.ScriptObjectMirror import and updated dependencies
Fix NullPointerException in JRubyAdapter.printStackTrace()
Fix bug in VcfImporterService where elasticsearch throws an exception when trying to flush after importing a large dataset
Added environment variable 'cluster.routing.allocation.disk.threshold_enabled' with value 'false' to ElasticsearchContainer to prevent Flink from failing due to disk space issues on the container.
Refactor ByteSelectiveStreamReader to reduce memory usage and improve performance by avoiding unnecessary allocations.
Refactor middleware execution logic to provide better debug logs
Refactor PropertiesEscapeUtil class to use the latest version of org.unbescape library.
Added a new Hazelcast instance to the cluster by invoking onPartition method of OperationServiceImpl
Improve nonce generation for BTCEBasePollingService class.
Improve performance of ParserTransformer by using a custom ClassLoader
Refactor: Move shutdown logic to new class for clarity and reusability
Fix issue with closing connection if already closed #<HASH>
Fix SurfaceLayerGL#destroy() to correctly set its own reference to SurfaceGL to null.
Fix issue with SimpleDateFormat and Date object initialization.
Update header retrieval method to handle null headers correctly
Update ECKey class to handle new DLSequence ASN.1 primitive correctly.
Added support for specifying roles in the `security-constraints` configuration section of the Undertow fraction.
Improve property evaluation by short-circuiting early.
Fix issue with FileSystemNotFoundException when creating a new FileSystem from URI in DefaultClassPathResourceLoader
Update PluginManagerInterceptor to allow preExecute() listeners to modify Mojo configuration before it is executed
Fix error when document contains 'www.metaescort.com'
Fix whitespace trimming in StoryState output stream. Remove existing glue and extra whitespace (spaces/tabs) from start of function.
Added assertion for int1 field in MockTest
Refactoring of BranchTemplateServiceImpl to use a new copy method with support for replacement function and policy
Added null check for output stream in Delombok#close(), fixes #42
Added a new try-catch block to handle any exceptions that may occur during the execution of the RepeatTestFilter.CURRENT_REPEAT_ACTION. The new catch block sets the value of CURRENT_REPEAT_ACTION back to null, ensuring that the repeats continue to run correctly even if an exception is thrown.
Fix unmapped place manager warning in PlaceRegistry#unmapPlaceManager(). The change includes adding a new line of code for logging the class name and OID of the place manager being unmapped.
Fix incorrect calculation of period between two dates for leap year and non-leap year scenarios. Fix #1234.
Optimized RegisteredLookupExtractionFn for better performance
Update ServerRemoveHandler to use RUNTIME stage instead of IMMEDIATE
Fix inconsistent length check for shares in `SecretSharing` constructor
Added support for converting camel case to human readable format using regex pattern matching and improved the performance of the isJreClass method by optimizing the regular expression.
Added LanguageTool menus and context menu interceptor. This commit refactors the code to use the new menus and context menu interceptor instead of the old approach.
Refactor KeyChainGroup constructor to use ArrayList instead of List for the chains field
Update LegacyAddress class to support new address types from SegWit
Fix exception handling in JUnitTestRunner to prevent unexpected crashes on iOS
Added support for AuthoritiesConstants in UserResource
Fix bug in InstructionSet.isDeclaration(), add null check for lvi variable.
Added `BootChargingStationCommand` class with `chargingStationId`, `protocol`, and `attributes` fields as well as `identityContext` field.
Refactor JsfHelper to support different types of datastores
Remove unnecessary code and improve readability
Added toolstrip at the top with zoom and save buttons
Added pre-crawling plugins for better plugin functionality and efficiency.
Added check for parent directory existence before attempting to create directories recursively.
Update BasicHttpClient to use a new version of HttpClient with improved performance and security.
Update DirectLogFetcherTest to use new writePkg method from PacketManager for writing a binlog dump request packet.
Added a delay before exiting the JVM to give some time for the user to connect to the daemon and start running tests
Update CollapsingTitleBar background color when title bar is expanded
Updated memory registration logic to handle negative object IDs and provide a more informative error message.
Refactor activity type property name to use consistent naming conventions
Improve performance of CampAST.emit() by replacing the StringWriter approach with a temporary replacement using toString().
Added a new test case for order by with MP subqueries.
Added support for serializing and deserializing common log fields (stacktrace) using Jackson annotations
Added URL protocol handling for HTTPS and FTP in BoxBrowser class.
Refactor CanvasTestBase to make it abstract
Update NotificationConsumer interface to support generics
Added support for validating columns with non-English characters in their names
Update DisposableHelper to use atomic reference for disposing Disposable instances
Fix a bug in Streams.java
Refactored Setup class to use new IoC container, JPA.bootstrap() and added bean scanning for annotated controllers
Update RxJavaObservableExecutionHook hook initialization
Refactoring of DRF.java file to shuffle rows and extract column values for variable importance calculation.
Fix row count mismatch in FindAllQuery by adding a secondary loop for fetching all rows and checking that the actual row count matches the expected value. Fixes #123.
Improved class loading performance by reusing ClassLoaders and implementing a faster findClass() method.
Added icon enrichment for Fabric8 Maven Plugin
Increased send and receive buffers for unicast socket to prevent retransmissions during large state transfer.
Update Android SDK class to support different platforms
Update error code for StaleElementReferenceException in AndroidServlet
Refactoring: updated ModelUtils class for new changes in OpenEngSB API
Update cookie expiration date format to match RFC 6265 compliance
[Remove unused constructor and log writer field] Reduce code size by removing unused constructor and log writer field
Refactor BrowserMobProxyTest to reduce unnecessary HTTP requests.
Fixes issue with text field focus line not resizing correctly on width change
Update test descriptions for context classes with specs and sub-context classes
Update Transaction Rule Builder to use the new `TransactionRule` class instead of `AuthorityRule`. ([#109](https://github.com/apache/shardingsphere/pull/109))
Update activejdbc configuration loading mechanism
Added support for local interfaces on DiameterStackMultiplexerAS7.
Update TerminateProvisionedProductRequest with random UUID for TerminateToken when empty string provided
Added phenotype validator and individual creation retries for offspring selection and population initialization.
Update files in eFaps using Multipart
Refreshing Kafka client configuration on property change
Refactor LazyPatternBy to make it more modular and testable by adding a new method 'toString()' to return the expression of the nested By instance and adding an override for 'getNested()' to return the nested By instance.
Update runtime properties for OMX app
Update query request version to use latest session sequence number.
Fix issue with generated IDs being negative
Added test for Jmol animation in CML files
Refactoring: Update ListItems to handle null collections and add @SuppressWarnings("unchecked") annotation for more robustness.
Refactor: Move indirect field access into intermediate class
Update test dependencies to use Jakarta API (#530)
Added HGVS code for SNV variants, updated samples data and files information in variant model
Added support for custom Maven repository location.
Refactoring of PathMatchingSimpleStorageResourcePatternResolver to improve performance by eliminating unnecessary resource existence checks.
Added a flag to track whether the site has received the first fragment task message. This allows the JoinProducer to know when it can start processing join fragments.
Added support for Jenkins 1.513 in ModelObjectWithChildren.java
Improve loss detection by using a more robust delay generation strategy
Fixes issue with parsing channel mentions containing large integer values.
Update dependency graph for referenced objects
Fix invalid code point handling in text encoding and decoding
Update RippleView class with new attributes and methods to support color customization and type selection.
Update Helper.java: remove obsolete methods and line endings handling
Improve performance of TSMeta#incrementCounter() by using callbacks for tree processing
Update DeploymentBuilder API to use RepositoryService instead of RuntimeService
Added test cases for checking none rows behavior with multiple queries and other threads.
Refactoring of SqlgStartStepBarrier class for better performance
Update List widget to support both OL and UL HTML tags
Fix race condition in Puller#handleMessage() that could result in messages being acknowledged twice.
Update low storage warning threshold in FileStoreLogFactory to 
Added click() method for WebElement and updated tests for toggle().
Update Gradle compatibility test matrix for Spring Boot project's Gradle plugin
Added support for SPARQL service description namespace
Fix memory calculation for large heap devices. (#3456)
Added support for prepared statements in LoadDataChange class
Added support for transient modifier in AdapterUtil
Added support for custom label in PerfMonCollector class
Update UsingToStringOrdering.java to use string representation instead of natural order of values.
Update CompilerConfiguration to allow for disabling global AST transformations using a comma-separated list of fully qualified class names
Added support for multiple graphs in parallel in TestMultiThread test class.
Fix for issue where receiver link was not being closed by the server when detached
Disable testCreateBucketedTable test case temporarily. The commit message should be: 'Disable testCreateBucketedTable test case temporarily'
Improve bitmap decoding performance by introducing a global lock for concurrent requests.
Update Predicates.java file with not predicate for different object types.
Fix: Clone method for TransitionSet class
Added null check for index column list in 'Index' class to handle cases where there are no columns specified in the index definition.
Update FiniteStateMachine to handle events with bodies
Fix issue with XML parser on CI machine. Remove Ignore annotation for tests.
Improve sound management by draining streams before shutting down lines.
Refactor LombokFileObjects to use the Javac8 FileManager when available.
Fix for LocalDate factory method ofYearDay to handle leap year correctly
Added deprecated constructor for UserProfile to support old codebases.
Remove unnecessary cast and simplify code in EngineTest class
Update StageTest.java
Refactoring of WAN REST test
Update ExclusiveBufferClaim to use byte instead of short for flags field
Added support for boolean user configuration parameter in OptimalCECPParameters class
Added support for rendering mode in ScheduleRenderer.
Update ShadowLooper to use a ThreadLocal instance for looperForThread to avoid race conditions.
Update AbstractFile toString method to include name of the file in path
Added new method 'append' to CmsClientReportWidget, which appends a string of HTML to the report content and scrolls the viewport to the bottom.
Fix error in initializing flag encoders for multi-profile profiles by ensuring that each profile has a unique name. This fixes an issue where multiple instances of the same profile would be created if multiple profiles were specified with the same name.
Added support for uploading files with forward slashes ('/') in their paths
Update ClientDelegatingFuture deserialization logic
Refactoring of Describe class to use SelenideElement for text representation in describe method
Added setter and getter methods for rulesByteArray field in SessionInfo class.
Refactor plan node CPU time calculation in `PlanPrinter`.\n\nThe CPU time for each plan node was previously calculated incorrectly. This commit fixes this issue by calculating the CPU time based on the `getPlanNodeCpuTime()` method instead of the `getPlanNodeScheduledTime()` method.
Added private constructor to prevent instantiation of CloudSpannerOAuthUtil class.
Update MySqlWriterCommands to use LIKE clause for table creation and truncate existing tables
Added support for resolving URIs from Paths and added a new method to return the URI of a resource.
Update SlaveReconnectionWork to use weak hash map for slave reconnection check
Refactoring ExtendedEmailPublisher to remove unused import for java.util.Random and minor improvements
Update WeaselProgram class for new Weasel mutator and selector. Refactor package name from io.jenetics.ext.engine to io.jenetics.ext.
Fix missing resources in ModelTest
Fix BCrypt hash comparison for constant-time equality checks.
Refactoring of DocumentGraphMapper class for better handling of ANNIS node and edge data.
Update DocGenTool to use JCommander for command line argument parsing.
Fix a bug in ProxyDefaultMethodHandle class by removing unused variable proxy and casting it to Object before calling INVOKE_DEFAULT_METHOD_HANDLE.invokeExact method.
Added support for underscored ISO codes in I18n
Fix type inference for generic methods
Update ClassFileWriter to support JDK7 and fix #630111 issue
Fix a potential security vulnerability in MulticastService by validating incoming JoinRequests based on their packet version.
Improve performance of DagManager by reducing unnecessary iterations through the running jobs list. Fixes #1234.
Remove shutdown hook registration for GraphDatabaseService.
Update PsLinkedin class to use LinkedIn API version 2.9 and add support for retrieving user's email address
Update WorkerOptions to use default value of workers = 1 when no value is provided.
Update Java code to skip specific log statements in formatter
Upgrade connection security restriction
Update JCA resource adapter information output for bidirectional and outbound resource adapters
Fix: Update CulpritsRecipientProvider to support newer AbstractBuild classes
Fix issue with disconnecting from voice channels by firing an AudioDisconnectEvent when no longer connected
Updated InstallRequest to read object from buffer with new buffer API.
Added support for Google Guava's Preconditions class in JoinMatcher
Improve performance of SourceCode#exactError by limiting the range of lines to analyze.
Fix for Windows compatibility issue in ChronicleMapBuilder.
Added support for filtering partitions based on constraints in getTableLayout(Session, TableHandle).
Update LakeBTC SSL URI and host name for correctness and compatibility with lakebtc.com
Added configuration for connection timeout and read timeout of FileDownloader.
Added a new method 'refresh' that calls ContextRefresher.refresh() and logs the result. Also changed the behavior of the original 'handleRefreshEvent' method to call the new method if the event is targeting this app, otherwise it will log a warning message. This change makes it possible for other applications to refresh their contexts as well.
Added error handling for RuntimeExceptions in ScheduledReporter#report.
Fix issue with counting line breaks in error message. Fix #1234.
Added a new method isIdElement(String) to check if an element of a path segment is an ID. Changed the implementation of doDetermineRestfulPath() to take advantage of this new method.
Update isPrivate method to ignore private methods that start with "_" and end with "__".
Added public access modifier to WebSocketSubscriptionTransport constructor to make it public
Enable/disable ViewPropertyDescriptor for new instances based on current view
Fix issue with empty filename causing FileNotFoundException. Add 'cancel' extra to intent for better handling of this scenario.
Fixes a bug in the onCreateAdapter method by properly initializing the `adapter` variable. Previously, the variable was not set correctly and the method did not work as expected.
Fix sleep() method in OCommandExecutorScript to avoid interruption of threads during execution
Added support for custom host identifiers with UUIDs. Added Jackson annotations to allow for JSON serialization and deserialization of these objects.
Added support for escaping extra characters using `extraEscapeChars` parameter in `CSVWriter#escape` method.
Added support for non-canonical base pairing and redundant base pairings in BasePairParameters class.
Added support for parsing double values as timestamps when parsing NMEA sentences.
Normalise and minimise automata for type checking
Added support for a new ArtifactType (#<HASH>) and added a getter method for it in SvnConnectorConfiguration.java (#<HASH>)'.
Refactoring of CommandManager to use TreeSet for all command names instead of HashSet
Update TypeDescription to use UNDEFINED instead of null for undefined variable reference
Added new exception classes for caching and file hash validation.
Added support for messenger_extensions and fallback_url in WebButton class
Added support for passing the parent node and pool as parameters when retrieving an item from the session data manager.
Fix logging level for found capabilities in `DiscoveryEntryStoreInMemory`. Fixes #1234.
Update AbstractRepository to use SpringSpecificationsExecutor for querying.
Remove unnecessary method clear() and documentation
Update ingress publication channel and stream ID using URI
Fix index name change check
Updated GBMTest class to use UCI-large dataset instead of COVID-19.
Added name property to GeoJsonObject class.
Update MediaDriver configuration options
Refactoring of ClientConnection class for better performance and reliability.
Added support for managing user properties in mgmt-users.properties files
Update Task class to register new tasks in TaskStateTracker after start time is set.
Fix a compiler bug by adding a comment to suppress the warning about closing a Closeable class loader.
Added support for GitHub display name and email fetching, fallbacks to id if not available
Refactored ByteBuf class to include readForRecycle method for recycling of byte buffer.
Added support for uploading files with comments
Fix error propagation when using FluxCircuitBreakerOperator with non-subscription exceptions
Update copyright year in RepackageMojo.java
Added try-finally block around while loop to ensure DBCursor is closed properly
Added documentation annotation to J2ObjCIncompatible class
[Added Meta constructor with parameters (key, value)]
Refactor ObjFunction class to use floating-point numbers for better precision and accuracy
Update SslTest to use TLS_DHE_RSA_WITH_AES_256_GCM_SHA384 instead of TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256 for SSL connection testing
Update GermanSpellerRule with new compound word forms and grammar rules.
Update unpackInt method in DataIO class to use DataInput interface instead of DataInputStream for improved performance.
Update DevelopmentPlugin to use a single VelocityEngine instance instead of multiple ones.
Update thread configuration for PlannerBenchmarkConfig class.
Added support for configuring slow and very slow invocation thresholds in milliseconds
Implemented OpsAgents and stats agent in VoltDB
Fix for issue #123: Improved performance by limiting path length in `getPathsOfLength()` method
Added support for a manual provider to display man pages.
Fix issue with unzipping and copying files on Android devices
Fix bug in UserExtensionsTest.java where empty extension list is returned instead of null
Update email address and subject line in rate me dialog fragment for customization purposes.
Added support for specifying multiple entity classes in constructor
Update S3ProxyHandler to support x-amz-acl header for multipart uploads
Remove unnecessary code from MapController.java
Added support for local variables in Xbase type computation. This change allows the type computer to handle local variables and their initializers during type resolution.
Added support for transforming Code.ForAll loops.
Added support for registering plugins in the `MainActivity`. Also added a debug view to display the version number.
Update Transaction class to support versioning and time locking
Fix showMenuItem method in JemmyDSL to work correctly with new String[] parameter.
Fix: Construct exception objects using constructors for better error handling and serialization compatibility
Fix issue with forward declarations not being added properly for private local types
Added warm-up for partitions in small cluster test
Added support for singleton scheduling option in the Scheduling module.
Added support for displaying culprits of failed builds in Dependency Graph
Removed unnecessary imports and reactor configuration in test class `WarJSFIntegrationTest`.
Fix scheduleAtFixedRate to scheduleWithFixedDelay in StreamSource for proper latency metrics
Optimized Selector: 
Fix a bug in TestUtil class
Added alt text to image editor for better accessibility support
Added node role field to Cluster class.
Update type arguments for bean definition builder
Added provisioning option when starting vagrant box (#1234).
Added support for emitting notifications in read-only context
Added check for mismatching number of fields in record schemas.
Fix error handling for REST calls in AbstractRemoteCommandObject
Added support for branding in DocbookBuilder using custom file names and common brand dirs.
Fix issue with PythonPlanStreamer when closing socket connection. Add null check for server object before calling close method.
Remove unused method 'increaseUsageCount()' from ClassDescriptor class. This method was not used in any commit and its removal does not impact the functionality of the application.
Added MaterializeSubscriber class to support materializing Flux streams using new Subscriber API.
Added support for monitoring changes in configuration files and automatically reloading the configuation.
Fix error handling in AbstractMappingStrategy when creating AgiScript instances
Update ProductRegisterEndPoint.java with a new author
Update PUBREC class to remove unnecessary constructor
Added test for large date format in Pact DSL JSON body class
Update CollapsePropertiesTest to use a custom coding convention
Added support for standalone mode and updated production environment detection logic to reflect it.
Added support for profile types in CQL retrieves and updated the behavior of retrieves to return the profile type rather than the base type. This change resolves inconsistencies introduced by the previous implementation.
Enable security features for gRPC server
Added support for IBM's GMT time zone ID
Optimize chunk size for classfile binary parsing for better load leveling
Update file upload logic for forced glossary and parallel corpus
Update NotifyTest.java to use PostgreSQL 9.0 or later protocol version
Refactor rejected notification listeners to remove expired ones efficiently.
Added support for Feature Requirements in assumeThat calls
Remove unnecessary strict mode policies
Updating SnapshotRestore sysproc to handle new json field names JSON_PATH and JSON_NONCE
Added support for querying artifacts by location in the ArtifactQuery class. Added a new constructor and getter method for the location field.
Update documentation for DiskLruCache.java
Update thread pool size for async image scaling
Fix: Update logger reference in FxmlLoadResult
Add support for parsing CADD scores
Added support for building all KieBases contained in a KieModule using a class filter.
Added code to clear up all attachments for a deployment unit after it has been stopped or undeployed.
Fix: removeAll method now returns 'changed' instead of 'removed'
Added a new test case to validate that schema validation using JsonSchemaRepository works correctly
Fix issue with null return value in LazilyLoadedCtor#construct, introduced by change in ScriptableObject.buildClassCtor(). Now returns null if the object has a static initializer.
Election: update candidate term id and transition to leader
Added option to disable summary generation using flag 'findbugs.noSummary'. Modified code to check for this flag and skip generating summary if necessary.
Refactor OrganizationDto class to remove redundant fields and improve code readability
Added support for waiting for Opera is idle and window load when changing the URL.
Refactor SwissKnife.java to use the JodaTime library for Date conversions.
Fix issue with calling 'inspect' on non-string value in Script.execute() method (#12345)
Added detection for partially constructed objects when invoking non-final, non-private methods from a constructor.
Update copyright year in AnalyzerPhase class. (#1234)
Fix corrupted hint name issue in HintedHandOffManager by updating getTableAndCFNames() method to account for potential array offset and length issues.
Fix issue with IFFImageWriter writing incorrect size in header chunk
Added support for setting path and description of a resource
Increased limit of open files from 8000 to 1024 to improve performance
Improve backpressure handling by not writing messages when channel is not writable and deferring them until the channel becomes writable again. Fixes #1234.
Fix invalid property values for sharding constants
Update AndPermission to support Context instead of Fragment
Added a test for thread acquired locks
Rename file operation in MasterInfo class
Added accessCheckCache field in Permit.Fake class for caching method lookup results.
Fix bug in PBDSegment#writeTruncatedEntry that causes incorrect behavior when compressedLength is larger than the original length of the buffer.
Fix memory leak in SqlStageExecution
Refactoring of ThriftClusterImpl to use ThriftUtils to convert Properties to CfDef
Added a check for operatorStats.isDone() before returning the last page from exchangeClient.getNextPage(). This ensures that we don't return any more data after all clients have completed.
Update TreeNodeLongClickListener interface to include return type for onLongClick method
Fix bug in ConfigException.ValidationFailed where an empty problem list was throwing a runtime exception
Update temp config file for integration tests
Refactor SubResourceSetting to use Java 8+ features, such as Optional and Stream API
Added PriorityResource class binding
Update request and response objects in filter chain for compatibility with newer Java versions
Fix issue with marking tokens outside of pattern in XML rule handler
Improved performance by reducing the number of threads used for generating requests from 4 to 20
Add support for generating primary key statements on non-Oracle databases
Improve term id management in Publication#incrementActiveTermId() method by updating next index and previous index accordingly
Added support for specifying a static factory method for generated classes. (#207)
Added check for operation mode before taking a snapshot.
Update JspcMojoClassLoader to use tools.jar for compilation
Added debug logging for sending messages
Added compareAndSet method for atomic sequence comparison and setting.
Added test for inserting a record with special characters (single quotes) in a column
Update PeriodicalRefreshStrategyTest to use atLeast instead of atLeastOnce for more accurate verification
Update AsyncExecute class to set environment variables for executing browser launch commands
Load instance fonts for search UI
Update initialStore method to handle methods with receivers correctly.
Update Kie Controller Swagger documentation for Business Central integration test
Update safe mode manager to use millisecond for time calculation
Added test cases for encoding special characters and tested text mode of PDF417 encoder
Added validation for timeslot duration and alm domain input fields in SseBuilder class. Timeslot duration must be set and higher than 30, and ALM domain must be a non-empty string.
Improve performance of NettyTransportTest by increasing timeouts
Fix access token handling in MPRestClient
Added support for including atomic attributes in VCF annotation output.
Refactoring of Task.java, removed unnecessary code and added comments
Added support for defining multiple cache managers in tests.
Added support for generic types in BeanSpec.java.
Fix performance state container test cases to not depend on empty map implementation.
Refactor layer state management to reduce memory usage
Fix incorrect type checking for repeated fields in ResolveExpressionTypesPass.
Added methods for retrieving policy rule URL and its ID.
Fix text size header in LogResource
Added onRequestPermissionsResult method to NativeScriptActivity class to support Android Marshmallow permissions
Added native implementation for Thread.interrupted() method.
Fix error handling for port conflict errors during rolling update operations. The error message is now more informative when the conflicting job was deployed manually or by a different deployment group.
Update max text length for anonymous users
Refactor: Deprecate unused methods in ContentMap.java.
Fix newline and tab character validation in RdfUtils#normalize() method.
Fix whitespace handling in XWikiXHTMLWhitespaceXMLFilter
